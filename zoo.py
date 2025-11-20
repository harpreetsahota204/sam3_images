"""
SAM3 Model for FiftyOne - Unified implementation supporting multiple operations.

Operations:
- concept_segmentation: Find ALL matching instances using text/visual prompts
- visual_segmentation: Segment SPECIFIC instances using interactive prompts
- automatic_segmentation: Generate all masks without prompts
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints
from fiftyone.utils.torch import GetItem

logger = logging.getLogger(__name__)


# Operations registry
SAM3_OPERATIONS = {
    "concept_segmentation": {
        "model_class": "Sam3Model",
        "processor_class": "Sam3Processor",
        "description": "Find ALL matching instances (text/visual concept)",
    },
    "visual_segmentation": {
        "model_class": "Sam3TrackerModel", 
        "processor_class": "Sam3TrackerProcessor",
        "description": "Segment SPECIFIC instances (interactive prompts)",
    },
    "automatic_segmentation": {
        "model_class": "Sam3TrackerModel",
        "processor_class": "Sam3TrackerProcessor",
        "description": "Automatic mask generation (no prompts)",
    }
}


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Sam3GetItem(GetItem):
    """GetItem transform for loading images and prompts for SAM3 operations."""
    
    def __init__(self, field_mapping=None, operation=None, prompt_field=None):
        # Set attributes BEFORE calling super().__init__
        # because parent init accesses required_keys which needs these
        self.operation = operation
        self.prompt_field = prompt_field
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        """Keys needed from each sample based on operation."""
        keys = ["filepath"]
        
        # Always include prompt_field if it's set (not None)
        # Parent GetItem handles validation and mapping
        if self.prompt_field is not None:
            keys.append("prompt_field")
        
        return keys
    
    def __call__(self, sample_dict):
        """
        Load image and extract prompt. Runs in DataLoader workers.
        
        Returns:
            Dict with 'image', 'original_size', and 'prompt'
        """
        # Load image
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        
        result = {
            'image': image,
            'original_size': image.size,  # (width, height)
            'prompt': None,
        }
        
        # Extract prompt using logical key "prompt_field"
        # field_mapping handles the actual dataset field mapping
        if "prompt_field" in sample_dict:
            result['prompt'] = sample_dict["prompt_field"]
        
        return result


class Sam3Model(Model, SupportsGetItem, TorchModelMixin):
    """
    Unified SAM3 model for FiftyOne with batching support.

    """
    
    def __init__(
        self,
        model_path: str = "facebook/sam3",
        operation: str = "concept_segmentation",
        prompt: str = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        points_mask_index: int = 0,
        auto_kwargs: dict = None,
        pooling_strategy: str = "mean",
        return_semantic_seg: bool = False,
        device: str = None,
        **kwargs
    ):
        """
        Initialize SAM3 model.
        
        Args:
            model_path: HuggingFace model ID or local path
            operation: One of: concept_segmentation, visual_segmentation, automatic_segmentation
            prompt: Default text prompt (for concept_segmentation)
            threshold: Confidence threshold for detections
            mask_threshold: Threshold for mask binarization
            points_mask_index: Which mask to use for point prompts (0=best, 1, 2)
            auto_kwargs: Additional kwargs for automatic mask generation
            pooling_strategy: Pooling strategy for embeddings (mean, max, cls)
            return_semantic_seg: Whether to include semantic segmentation mask (concept_segmentation only)
            device: Device to use (cuda/cpu/mps, None for auto)
        """
        # Initialize base classes - SupportsGetItem NOT SamplesMixin!
        SupportsGetItem.__init__(self)
        
        # Required flags for FiftyOne batching
        self._preprocess = False
        self._fields = {}
        
        # Validate operation
        if operation not in SAM3_OPERATIONS:
            raise ValueError(
                f"Invalid operation: {operation}. "
                f"Must be one of {list(SAM3_OPERATIONS.keys())}"
            )
        self._operation = operation
        
        # Store configuration
        self.model_path = model_path
        self.default_prompt = prompt
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.points_mask_index = points_mask_index
        self.auto_kwargs = auto_kwargs or {}
        self.pooling_strategy = pooling_strategy
        self.return_semantic_seg = return_semantic_seg
        
        # Embeddings support
        self._last_computed_embeddings = None
        
        # Setup device
        if device is None:
            device = get_device()
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self):
        """Load appropriate model and processor based on operation."""
        from transformers import (
            Sam3Model, Sam3Processor,
            Sam3TrackerModel, Sam3TrackerProcessor
        )
                
        logger.info(f"Loading SAM3 for operation: {self._operation}")
        
        if self._operation == "concept_segmentation":
            # Concept segmentation: text/visual prompts → all matching instances
            self.model = Sam3Model.from_pretrained(self.model_path).to(self.device)
            self.processor = Sam3Processor.from_pretrained(self.model_path)
        
        else:  # visual_segmentation or automatic_segmentation
            # Visual segmentation: interactive prompts → specific instances
            # Automatic segmentation: point grid sampling
            self.model = Sam3TrackerModel.from_pretrained(self.model_path).to(self.device)
            self.processor = Sam3TrackerProcessor.from_pretrained(self.model_path)
        
        self.model.eval()
        logger.info("SAM3 model loaded successfully")
    
    # ==================== OPERATION PROPERTIES ====================
    
    @property
    def operation(self):
        """Current operation mode."""
        return self._operation
    
    @operation.setter
    def operation(self, value):
        """Change operation (reloads model if needed)."""
        if value not in SAM3_OPERATIONS:
            raise ValueError(
                f"Invalid operation: {value}. "
                f"Must be one of {list(SAM3_OPERATIONS.keys())}"
            )
        if value != self._operation:
            self._operation = value
            self._load_model()
    
    @property
    def prompt(self):
        """Default text prompt."""
        return self.default_prompt
    
    @prompt.setter
    def prompt(self, value):
        """Set default text prompt."""
        self.default_prompt = value
    
    # ==================== FROM Model (REQUIRED) ====================
    
    @property
    def media_type(self):
        return "image"
    
    @property
    def transforms(self):
        return None  # GetItem handles preprocessing
    
    @property
    def preprocess(self):
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        self._preprocess = value
    
    @property
    def ragged_batches(self):
        """MUST be False to enable batching!"""
        return False
    
    @property
    def needs_fields(self):
        """Dict mapping model keys to dataset field names."""
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    # ==================== FROM SupportsGetItem (REQUIRED) ====================
    
    def build_get_item(self, field_mapping=None):
        """Build GetItem transform for data loading."""
        # Extract prompt field - check both field_mapping and needs_fields
        prompt_field = None
        if field_mapping and "prompt_field" in field_mapping:
            prompt_field = field_mapping["prompt_field"]
        elif "prompt_field" in self._fields:
            prompt_field = self._fields["prompt_field"]
        
        return Sam3GetItem(
            field_mapping=field_mapping,
            operation=self._operation,
            prompt_field=prompt_field
        )
    
    # ==================== FROM TorchModelMixin (REQUIRED) ====================
    
    @property
    def has_collate_fn(self):
        """Custom collation needed for variable-size images."""
        return True
    
    @property
    def collate_fn(self):
        """Return batch as-is (list of dicts) without stacking."""
        @staticmethod
        def identity_collate(batch):
            return batch
        return identity_collate
    
    # ==================== CORE INFERENCE ====================
    
    def predict(self, image, sample=None):
        """
        Single-sample inference.
        
        Args:
            image: PIL Image, numpy array, or filepath
            sample: Optional FiftyOne sample for extracting prompts
            
        Returns:
            fo.Detections
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Extract prompt from sample if available
        prompt = None
        if sample is not None and "prompt_field" in self._fields:
            field_name = self._fields["prompt_field"]
            if sample.has_field(field_name):
                prompt = sample.get_field(field_name)
        
        # Create batch item
        batch_item = {
            'image': image,
            'original_size': image.size,
            'prompt': prompt,
        }
        
        # Use batch inference
        results = self.predict_all([batch_item])
        return results[0]
    
    def predict_all(self, batch, preprocess=None):
        """
        Batch inference - routes to operation-specific method.
        
        Args:
            batch: List of dicts from GetItem with 'image', 'original_size', 'prompt'
            preprocess: Whether to apply preprocessing (unused, handled by GetItem)
            
        Returns:
            List of fo.Detections (one per image)
        """
        if not batch:
            return []
        
        # Extract prompts from batch
        prompts = [item.get('prompt') for item in batch]
        
        # Route to operation-specific method
        if self._operation == "concept_segmentation":
            return self._predict_concept_segmentation(batch, prompts)
        elif self._operation == "visual_segmentation":
            return self._predict_visual_segmentation(batch, prompts)
        elif self._operation == "automatic_segmentation":
            return self._predict_automatic_segmentation(batch)
        else:
            raise ValueError(f"Unknown operation: {self._operation}")
    
    # ==================== OPERATION IMPLEMENTATIONS ====================
    
    def _predict_concept_segmentation(self, batch, prompts):
        """
        Concept segmentation: Find ALL matching instances using text prompts.
        
        When prompt is a list, runs inference for each concept and merges results.
        
        Args:
            batch: List of dicts with 'image' and 'original_size'
            prompts: List of text prompts (str, list, or None)
        """
        # Check if any prompt is a list (multi-concept search)
        has_multi_concept = any(isinstance(p, list) for p in prompts)
        
        if has_multi_concept:
            return self._predict_multi_concept(batch, prompts)
        
        images = [item['image'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Parse text prompts
        text_prompts = []
        
        for prompt in prompts:
            if isinstance(prompt, str):
                text_prompts.append(prompt)
            
            elif prompt is None:
                # Use default
                default = self.default_prompt
                if isinstance(default, list):
                    logger.info(f"Default prompt is list {default}. Using multi-concept search.")
                    return self._predict_multi_concept(batch, prompts)
                
                text_prompts.append(default if default else "object")
            
            else:
                raise TypeError(
                    f"concept_segmentation expects str or list text prompts, "
                    f"got {type(prompt).__name__}. "
                    f"Use visual_segmentation for Detections/Keypoints prompts."
                )
        
        # Run SAM3 inference
        inputs = self.processor(images=images, text=text_prompts, return_tensors="pt").to(self.device)
        
        logger.info(f"Processing {len(images)} images with text prompts: {text_prompts[:2]}...")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process to get instance segmentation
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        
        logger.info(f"Found {sum(len(r['masks']) for r in results)} total instances")
        
        # Extract semantic segmentation if requested
        semantic_masks = None
        if self.return_semantic_seg:
            semantic_masks = self._extract_semantic_segmentation(
                outputs.semantic_seg,
                original_sizes
            )
        
        # Convert to FiftyOne format
        fo_results = []
        for i, result in enumerate(results):
            label = text_prompts[i]
            
            if len(result['masks']) > 0:
                detections = self._sam3_to_detections(
                    masks=result['masks'],
                    boxes=result['boxes'],
                    scores=result['scores'],
                    width=original_sizes[i][0],
                    height=original_sizes[i][1],
                    label=label
                )
            else:
                detections = Detections()
            
            if semantic_masks is not None:
                detections.semantic_mask = semantic_masks[i]
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_multi_concept(self, batch, prompts):
        """
        Run concept segmentation for multiple concepts per image.
        
        When a prompt is a list, we run inference once per concept and merge results.
        This allows finding multiple object types in each image.
        
        Args:
            batch: List of dicts with 'image' and 'original_size'
            prompts: List of prompts (some may be lists)
        """
        images = [item['image'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Normalize all prompts to lists of concepts
        concepts_per_image = []
        for prompt in prompts:
            if isinstance(prompt, list):
                concepts_per_image.append(prompt)
            elif isinstance(prompt, str):
                concepts_per_image.append([prompt])
            else:
                # None - use default
                default = self.default_prompt
                concepts_per_image.append(
                    default if isinstance(default, list) else [default or "object"]
                )
        
        # Get unique concepts across all images
        all_concepts = set()
        for concepts in concepts_per_image:
            all_concepts.update(concepts)
        all_concepts = sorted(list(all_concepts))
        
        logger.info(f"Running multi-concept search for {len(all_concepts)} concepts: {all_concepts}")
        
        # Run inference once per concept
        results_per_concept = {}
        for concept in all_concepts:
            # Create batch with same concept for all images
            concept_prompts = [concept] * len(images)
            
            # Run inference
            inputs = self.processor(
                images=images,
                text=concept_prompts,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            
            results_per_concept[concept] = results
        
        # Merge results per image
        fo_results = []
        for i in range(len(images)):
            all_detections = []
            w, h = original_sizes[i]
            
            # Collect detections from each concept this image requested
            for concept in concepts_per_image[i]:
                if concept in results_per_concept:
                    result = results_per_concept[concept][i]
                    
                    if len(result['masks']) > 0:
                        # Convert to FiftyOne detections
                        concept_dets = self._sam3_to_detections(
                            masks=result['masks'],
                            boxes=result['boxes'],
                            scores=result['scores'],
                            width=w,
                            height=h,
                            label=concept
                        )
                        all_detections.extend(concept_dets.detections)
            
            fo_results.append(Detections(detections=all_detections))
        
        logger.info(f"Multi-concept search complete. Total instances: {sum(len(r.detections) for r in fo_results)}")
        
        return fo_results
    
    def _predict_visual_segmentation(self, batch, prompts):
        """
        Visual segmentation: Segment SPECIFIC instances.
        Supports box prompts (Detections) or point prompts (Keypoints).
        
        Args:
            batch: List of dicts with 'image' and 'original_size'
            prompts: List of prompts (Detections, Keypoints, or None)
        """
        # Detect prompt type from first non-None prompt
        # All prompts in batch will be same type (from same field)
        prompt_type = None
        for prompt in prompts:
            if prompt is not None:
                if isinstance(prompt, Detections):
                    prompt_type = "boxes"
                elif isinstance(prompt, Keypoints):
                    prompt_type = "points"
                else:
                    raise TypeError(
                        f"visual_segmentation expects Detections or Keypoints prompts, "
                        f"got {type(prompt).__name__}"
                    )
                break
        
        # Route based on prompt type
        if prompt_type == "boxes":
            return self._predict_visual_boxes(batch, prompts)
        elif prompt_type == "points":
            return self._predict_visual_points(batch, prompts)
        else:
            # No prompts - return empty detections
            return [Detections() for _ in batch]
    
    def _predict_visual_boxes(self, batch, box_prompts):
        """
        Process visual segmentation with box prompts.
        
        Note: SAM3 Tracker cannot batch images with different numbers of boxes.
        Falls back to sequential processing when box counts differ.
        """
        images = [item['image'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Count boxes per image and check if we can batch
        box_counts = [len(p.detections) if p and len(p.detections) > 0 else 0 for p in box_prompts]
        
        if len(set(box_counts)) > 1:
            logger.warning(
                f"Box counts vary {box_counts}. "
                f"SAM3 Tracker cannot batch variable counts. Processing sequentially."
            )
            return self._predict_visual_boxes_sequential(batch, box_prompts)
        
        # Convert FO boxes to SAM3 format: [batch][num_boxes][4]
        all_boxes = []
        all_labels = []
        
        for prompts, (w, h) in zip(box_prompts, original_sizes):
            if prompts and len(prompts.detections) > 0:
                # Convert all boxes for this image
                boxes_for_image = [
                    [det.bounding_box[0] * w,
                     det.bounding_box[1] * h,
                     (det.bounding_box[0] + det.bounding_box[2]) * w,
                     (det.bounding_box[1] + det.bounding_box[3]) * h]
                    for det in prompts.detections
                ]
                labels_for_image = [det.label for det in prompts.detections]
                
                all_boxes.append(boxes_for_image)
                all_labels.append(labels_for_image)
            else:
                # Empty prompts - dummy box
                all_boxes.append([[0, 0, 100, 100]])
                all_labels.append(["object"])
        
        # Process with SAM3 Tracker
        inputs = self.processor(
            images=images,
            input_boxes=all_boxes,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Post-process and extract scores
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
        iou_scores = outputs.iou_scores.cpu() if hasattr(outputs, 'iou_scores') else None
        
        # Convert to FiftyOne Detections
        fo_results = []
        for i, (mask, labels, (w, h)) in enumerate(zip(masks, all_labels, original_sizes)):
            if box_prompts[i] and len(box_prompts[i].detections) > 0:
                # Convert boxes to xyxy (reuse from all_boxes)
                boxes_xyxy = np.array(all_boxes[i])
                image_scores = iou_scores[i] if iou_scores is not None else None
                
                detections = self._tracker_to_detections(
                    masks=mask,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    width=w,
                    height=h,
                    scores=image_scores
                )
            else:
                detections = Detections()
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_visual_boxes_sequential(self, batch, box_prompts):
        """Process visual boxes one-by-one when batch has variable box counts."""
        fo_results = []
        
        for item, prompts in zip(batch, box_prompts):
            image = item['image']
            w, h = item['original_size']
            
            if not prompts or len(prompts.detections) == 0:
                fo_results.append(Detections())
                continue
            
            # Convert boxes for single image
            boxes_for_image = []
            labels_for_image = []
            
            for det in prompts.detections:
                rel_x, rel_y, rel_w, rel_h = det.bounding_box
                x1 = rel_x * w
                y1 = rel_y * h
                x2 = (rel_x + rel_w) * w
                y2 = (rel_y + rel_h) * h
                
                boxes_for_image.append([x1, y1, x2, y2])
                labels_for_image.append(det.label)
            
            # Process single image
            inputs = self.processor(
                images=image,
                input_boxes=[boxes_for_image],  # Single image: [num_boxes][4]
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            
            # Post-process (returns tensor)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0]
            
            # Extract IoU scores if available
            iou_scores = outputs.iou_scores.cpu()[0] if hasattr(outputs, 'iou_scores') else None
            
            # Convert to FiftyOne
            boxes_xyxy = np.array([
                [det.bounding_box[0] * w,
                 det.bounding_box[1] * h,
                 (det.bounding_box[0] + det.bounding_box[2]) * w,
                 (det.bounding_box[1] + det.bounding_box[3]) * h]
                for det in prompts.detections
            ])
            
            detections = self._tracker_to_detections(
                masks=masks,
                boxes_xyxy=boxes_xyxy,
                labels=labels_for_image,
                width=w,
                height=h,
                scores=iou_scores
            )
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_visual_points(self, batch, point_prompts):
        """
        Process visual segmentation with point prompts.
        
        Note: SAM3 Tracker cannot batch images with different numbers of keypoints.
        Falls back to sequential processing when keypoint counts differ.
        """
        images = [item['image'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Count keypoints per image and check if we can batch
        kp_counts = [len(p.keypoints) if p and len(p.keypoints) > 0 else 0 for p in point_prompts]
        
        if len(set(kp_counts)) > 1:
            logger.warning(
                f"Keypoint counts vary {kp_counts}. "
                f"SAM3 Tracker cannot batch variable counts. Processing sequentially."
            )
            return self._predict_visual_points_sequential(batch, point_prompts)
        
        # Convert FO keypoints to SAM3 format
        all_points = []
        all_labels = []
        all_label_names = []
        
        for prompts, (w, h) in zip(point_prompts, original_sizes):
            if prompts and len(prompts.keypoints) > 0:
                points_for_image = []
                labels_for_image = []
                names_for_image = []
                
                for kp in prompts.keypoints:
                    # Convert all points in this keypoint to absolute coords
                    kp_points = [[pt[0] * w, pt[1] * h] for pt in kp.points]
                    kp_labels = [1] * len(kp.points)  # All positive
                    
                    points_for_image.append([kp_points])
                    labels_for_image.append([kp_labels])
                    names_for_image.append(kp.label)
                
                all_points.append(points_for_image)
                all_labels.append(labels_for_image)
                all_label_names.append(names_for_image)
            else:
                # Empty prompts - dummy point
                all_points.append([[[100, 100]]])
                all_labels.append([[[1]]])
                all_label_names.append(["object"])
        
        # Process with SAM3 Tracker
        inputs = self.processor(
            images=images,
            input_points=all_points,
            input_labels=all_labels,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Post-process and extract scores
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
        iou_scores = outputs.iou_scores.cpu() if hasattr(outputs, 'iou_scores') else None
        
        # Convert to FiftyOne Detections
        fo_results = []
        for i, (mask, names, (w, h)) in enumerate(zip(masks, all_label_names, original_sizes)):
            if point_prompts[i] and len(point_prompts[i].keypoints) > 0:
                boxes_xyxy = self._masks_to_boxes(mask, self.points_mask_index)
                image_scores = iou_scores[i] if iou_scores is not None else None
                
                detections = self._tracker_to_detections(
                    masks=mask,
                    boxes_xyxy=boxes_xyxy,
                    labels=names,
                    width=w,
                    height=h,
                    mask_index=self.points_mask_index,
                    scores=image_scores
                )
            else:
                detections = Detections()
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_visual_points_sequential(self, batch, point_prompts):
        """Process visual points one-by-one when batch has variable keypoint counts."""
        fo_results = []
        
        for item, prompts in zip(batch, point_prompts):
            image = item['image']
            w, h = item['original_size']
            
            if not prompts or len(prompts.keypoints) == 0:
                fo_results.append(Detections())
                continue
            
            # Convert points for single image
            points_for_image = []
            labels_for_image = []
            names_for_image = []
            
            for kp in prompts.keypoints:
                kp_points = []
                kp_labels = []
                
                for point in kp.points:
                    rel_x, rel_y = point
                    abs_x = rel_x * w
                    abs_y = rel_y * h
                    kp_points.append([abs_x, abs_y])
                    kp_labels.append(1)
                
                points_for_image.append([kp_points])
                labels_for_image.append([kp_labels])
                names_for_image.append(kp.label)
            
            # Process single image
            inputs = self.processor(
                images=image,
                input_points=points_for_image,
                input_labels=labels_for_image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            
            # Post-process (returns tensor)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0]
            
            # Extract IoU scores if available
            iou_scores = outputs.iou_scores.cpu()[0] if hasattr(outputs, 'iou_scores') else None
            
            # Extract boxes from masks
            boxes_xyxy = self._masks_to_boxes(masks, self.points_mask_index)
            
            # Convert to FiftyOne
            detections = self._tracker_to_detections(
                masks=masks,
                boxes_xyxy=boxes_xyxy,
                labels=names_for_image,
                width=w,
                height=h,
                mask_index=self.points_mask_index,
                scores=iou_scores
            )
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_automatic_segmentation(self, batch):
        """
        Automatic segmentation: Generate all masks without prompts.
        Uses point grid sampling for comprehensive mask generation.
        """
        fo_results = []
        
        for item in batch:
            image = item['image']
            w, h = item['original_size']
            
            # Generate automatic masks for this image
            detections = self._generate_automatic_masks(image, w, h)
            fo_results.append(detections)
        
        return fo_results
    
    def _generate_automatic_masks(self, image, width, height):
        """
        Generate automatic masks by sampling a grid of points.
        
        Includes quality filtering and deduplication for cleaner results.
        
        Args:
            image: PIL Image
            width: Image width
            height: Image height
            
        Returns:
            fo.Detections with filtered and deduplicated masks
        """
        # Get parameters
        points_per_side = self.auto_kwargs.get("points_per_side", 16)
        points_per_batch = self.auto_kwargs.get("points_per_batch", 256)
        quality_threshold = self.auto_kwargs.get("quality_threshold", 0.8)
        iou_threshold = self.auto_kwargs.get("iou_threshold", 0.85)
        max_masks = self.auto_kwargs.get("max_masks", None)
        
        # Generate point grid
        point_grid_x = np.linspace(0, width-1, points_per_side)
        point_grid_y = np.linspace(0, height-1, points_per_side)
        
        all_points = []
        for y in point_grid_y:
            for x in point_grid_x:
                all_points.append([x, y])
        
        # Process points in batches
        all_masks = []
        all_scores = []
        
        for i in range(0, len(all_points), points_per_batch):
            batch_points = all_points[i:i+points_per_batch]
            
            # Format: [batch=1][num_points][1 point][coords]
            input_points = [[[[px, py]] for px, py in batch_points]]
            input_labels = [[[1] for _ in batch_points]]
            
            # Process batch
            inputs = self.processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            
            # Post-process
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0]
            
            # Extract IoU scores
            iou_scores = outputs.iou_scores.cpu()[0] if hasattr(outputs, 'iou_scores') else None
            
            # Take best mask for each point (index 0 = highest quality)
            # masks shape: [num_points, 3, H, W]
            for j in range(len(batch_points)):
                all_masks.append(masks[j, 0].cpu().numpy())
                
                if iou_scores is not None:
                    # iou_scores shape: [num_points, 3]
                    all_scores.append(float(iou_scores[j, 0]))
        
        # Filter by quality
        if all_scores:
            filtered_masks = []
            filtered_scores = []
            
            for mask, score in zip(all_masks, all_scores):
                if score >= quality_threshold:
                    filtered_masks.append(mask)
                    filtered_scores.append(score)
            
            logger.info(f"Quality filter: {len(all_masks)} → {len(filtered_masks)} masks (threshold={quality_threshold})")
        else:
            filtered_masks = all_masks
            filtered_scores = all_scores
        
        # Deduplicate using NMS-style IoU filtering
        if len(filtered_masks) > 1:
            keep_indices = self._nms_masks(filtered_masks, filtered_scores, iou_threshold)
            filtered_masks = [filtered_masks[i] for i in keep_indices]
            filtered_scores = [filtered_scores[i] for i in keep_indices] if filtered_scores else None
            
            logger.info(f"Deduplication: {len(all_masks)} → {len(filtered_masks)} masks (IoU threshold={iou_threshold})")
        
        # Limit to max_masks if specified
        if max_masks and len(filtered_masks) > max_masks:
            # Sort by score and take top N
            if filtered_scores:
                sorted_indices = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)
                filtered_masks = [filtered_masks[i] for i in sorted_indices[:max_masks]]
                filtered_scores = [filtered_scores[i] for i in sorted_indices[:max_masks]]
            else:
                filtered_masks = filtered_masks[:max_masks]
            
            logger.info(f"Limited to top {max_masks} masks by quality")
        
        # Convert to FiftyOne Detections
        detections = []
        scores_list = filtered_scores if filtered_scores else [None] * len(filtered_masks)
        
        for idx, (mask, score) in enumerate(zip(filtered_masks, scores_list)):
            # Extract bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                box_xyxy = np.array([x1, y1, x2, y2])
                
                detection = self._create_detection(
                    mask=mask,
                    box_xyxy=box_xyxy,
                    label=f"object_{idx}",  # Bake index into label for clearer visualization
                    width=width,
                    height=height,
                    index=idx,
                    confidence=score
                )
                detections.append(detection)
        
        return Detections(detections=detections)
    
    def _nms_masks(self, masks, scores, iou_threshold):
        """
        Non-maximum suppression for masks based on IoU overlap.
        Uses bbox pre-filtering for 10-20x speedup.
        
        Args:
            masks: List of numpy masks [H, W]
            scores: List of confidence scores (higher is better)
            iou_threshold: IoU threshold for considering masks duplicates
            
        Returns:
            List of indices to keep
        """
        if not scores:
            return list(range(len(masks)))
        
        # Pre-compute bboxes for fast overlap checks
        bboxes = [self._mask_to_bbox(mask) for mask in masks]
        
        # Sort by score (highest first)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        keep = []
        for idx in sorted_indices:
            is_duplicate = False
            
            for keep_idx in keep:
                # Fast bbox IoU check first (10-50x faster than mask IoU)
                bbox_iou = self._bbox_iou(bboxes[idx], bboxes[keep_idx])
                
                # Only compute expensive mask IoU if bboxes overlap
                if bbox_iou > iou_threshold * 0.5:
                    mask_iou = self._compute_mask_iou(masks[idx], masks[keep_idx])
                    
                    if mask_iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                keep.append(idx)
        
        return keep
    
    def _mask_to_bbox(self, mask):
        """
        Extract bounding box from mask.
        
        Args:
            mask: Binary mask [H, W]
            
        Returns:
            Tuple (x1, y1, x2, y2) in absolute coordinates
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            return (x1, y1, x2, y2)
        
        return (0, 0, 1, 1)
    
    def _bbox_iou(self, box1, box2):
        """
        Compute IoU between two bounding boxes (fast).
        
        Args:
            box1: Tuple (x1, y1, x2, y2)
            box2: Tuple (x1, y1, x2, y2)
            
        Returns:
            float: IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # No overlap
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Compute areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_mask_iou(self, mask1, mask2):
        """
        Compute IoU between two binary masks.
        
        Args:
            mask1: Binary mask [H, W]
            mask2: Binary mask [H, W]
            
        Returns:
            float: IoU score
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    # ==================== EMBEDDINGS SUPPORT ====================
    
    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings."""
        return True
    
    def embed(self, item):
        """
        Embed a single image using SAM3 vision encoder.
        
        Args:
            item: Dict from GetItem, image path string, or FiftyOne sample/reader object
            
        Returns:
            numpy array: 1D embedding vector of shape (1024,)
        """
        # Load image based on type
        if isinstance(item, dict):
            image = item['image']  # From GetItem DataLoader
        elif isinstance(item, str):
            image = Image.open(item).convert("RGB")
        elif isinstance(item, Image.Image):
            image = item  # Already a PIL Image
        else:
            # Try to get filepath from object
            filepath = getattr(item, 'filepath', None) or getattr(item, 'path', None)
            if filepath:
                image = Image.open(filepath).convert("RGB")
            else:
                raise TypeError(f"Unsupported item type: {type(item)}")
        
        return self._extract_visual_embeddings(image)
    
    def embed_all(self, items):
        """
        Embed multiple images.
        
        Args:
            items: List of image paths or FiftyOne samples/readers
            
        Returns:
            numpy array: 2D array of shape (num_items, 1024)
        """
        embeddings = []
        
        for item in items:
            embedding = self.embed(item)
            embeddings.append(embedding)
        
        embeddings = np.stack(embeddings, axis=0)
        
        # Cache for get_embeddings()
        self._last_computed_embeddings = embeddings
        
        return embeddings
    
    def get_embeddings(self):
        """
        Return cached embeddings from last computation.
        
        Returns:
            numpy array: Last computed embeddings or None
        """
        return self._last_computed_embeddings
    
    def _extract_visual_embeddings(self, image):
        """
        Extract visual embeddings from SAM3 vision encoder.
        
        Text prompts do not affect embeddings - vision encoder processes
        images independently of text.
        
        Args:
            image: PIL Image
            
        Returns:
            numpy array: 1D embedding vector of shape (1024,)
        """
        # Process image (text is irrelevant for embeddings)
        inputs = self.processor(
            images=image,
            text="",  # Empty text has no effect on vision encoder
            return_tensors="pt"
        ).to(self.device)
        
        # Extract vision features
        with torch.no_grad():
            # Get last hidden state from vision encoder
            # Shape: [1, 5184, 1024] - 5184 visual tokens, 1024-dim each
            hidden_states = self.model.vision_encoder(
                inputs['pixel_values']
            ).last_hidden_state
        
        # Pool sequence to fixed dimension
        embedding = self._pool_embeddings(hidden_states)
        
        # Normalize for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().float().numpy()[0]
    
    def _pool_embeddings(self, hidden_states):
        """
        Pool sequence of visual tokens to fixed-size embedding.
        
        Args:
            hidden_states: Tensor of shape [1, 5184, 1024]
            
        Returns:
            Tensor of shape [1, 1024]
        """
        if self.pooling_strategy == "mean":
            # Average across all visual tokens
            pooled = torch.mean(hidden_states, dim=1)
        
        elif self.pooling_strategy == "max":
            # Max across all visual tokens
            pooled = torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_strategy == "cls":
            # Use first token (if it's a special CLS token)
            pooled = hidden_states[:, 0, :]
        
        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.pooling_strategy}. "
                f"Must be one of: mean, max, cls"
            )
        
        return pooled
    
    # ==================== CONVERSION UTILITIES ====================
    
    def _create_detection(
        self,
        mask: np.ndarray,
        box_xyxy: np.ndarray,
        label: str,
        width: int,
        height: int,
        index: int,
        confidence: Optional[float] = None
    ) -> Detection:
        """
        Create a single FiftyOne Detection from mask and box.
        
        Args:
            mask: [H, W] binary mask at original resolution
            box_xyxy: [4] bounding box in xyxy absolute coordinates
            label: Object class label
            width: Image width
            height: Image height
            index: Instance index for tracking
            confidence: Optional confidence score
            
        Returns:
            fo.Detection
        """
        x1, y1, x2, y2 = box_xyxy
        
        # Convert absolute xyxy to relative [x, y, width, height]
        rel_bbox = [
            x1 / width,
            y1 / height,
            (x2 - x1) / width,
            (y2 - y1) / height
        ]
        
        # Crop mask to bounding box
        y1_int, y2_int = int(round(y1)), int(round(y2))
        x1_int, x2_int = int(round(x1)), int(round(x2))
        cropped_mask = mask[y1_int:y2_int, x1_int:x2_int]
        
        return Detection(
            label=label,
            bounding_box=rel_bbox,
            mask=cropped_mask,
            confidence=confidence,
            index=index
        )
    
    def _extract_semantic_segmentation(
        self,
        semantic_seg: torch.Tensor,
        original_sizes: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """
        Extract and resize semantic segmentation masks to original image sizes.
        
        Args:
            semantic_seg: Tensor of shape [batch, 1, H, W]
            original_sizes: List of (width, height) tuples
            
        Returns:
            List of numpy arrays, one per image, at original resolution
        """
        
        
        semantic_masks = []
        
        # semantic_seg shape: [batch, 1, H, W]
        for i, (w, h) in enumerate(original_sizes):
            # Get mask for this image: [1, H, W]
            mask = semantic_seg[i]
            
            # Resize to original image size
            # Input: [1, H, W], need [1, 1, H, W] for interpolate
            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 1, H, W]
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to numpy and remove batch/channel dims: [H, W]
            mask_np = mask_resized.squeeze().cpu().numpy()
            
            # Threshold to binary (values are logits)
            mask_binary = (mask_np > 0).astype(np.uint8)
            
            semantic_masks.append(mask_binary)
        
        return semantic_masks
    
    def _sam3_to_detections(
        self,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        width: int,
        height: int,
        label: str
    ) -> Detections:
        """
        Convert SAM3 concept segmentation output to FiftyOne Detections.
        
        Args:
            masks: [N, H, W] binary masks at original resolution
            boxes: [N, 4] bounding boxes in xyxy absolute coordinates
            scores: [N] confidence scores
            width: Original image width
            height: Original image height
            label: Object class label
            
        Returns:
            fo.Detections with index attribute for instance tracking
        """
        # Convert to numpy
        masks_np = masks.cpu().numpy()
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        # Create detections using helper
        detections = [
            self._create_detection(
                mask=masks_np[i],
                box_xyxy=boxes_np[i],
                label=label,
                width=width,
                height=height,
                index=i,
                confidence=float(scores_np[i])
            )
            for i in range(len(boxes_np))
        ]
        
        return Detections(detections=detections)
    
    def _tracker_to_detections(
        self,
        masks: torch.Tensor,
        boxes_xyxy: np.ndarray,
        labels: List[str],
        width: int,
        height: int,
        mask_index: int = 0,
        scores: Optional[torch.Tensor] = None
    ) -> Detections:
        """
        Convert SAM3 Tracker output to FiftyOne Detections.
        
        Args:
            masks: [N, num_masks, H, W] or [N, H, W]
            boxes_xyxy: [N, 4] bounding boxes in xyxy absolute coordinates
            labels: List of N label strings
            width: Original image width
            height: Original image height
            mask_index: Which mask to use if multiple (0=best quality)
            scores: Optional [N] or [N, num_masks] confidence scores
            
        Returns:
            fo.Detections
        """
        masks_np = masks.cpu().numpy()
        scores_np = scores.cpu().numpy() if scores is not None else None
        
        # Select mask based on dimensions
        if masks_np.ndim == 4:
            selected_masks = masks_np[:, mask_index]  # [N, H, W]
        else:
            selected_masks = masks_np
        
        # Extract confidence scores if available
        confidences = []
        if scores_np is not None:
            for i in range(len(boxes_xyxy)):
                if scores_np.ndim == 2:
                    confidences.append(float(scores_np[i, mask_index]))
                else:
                    confidences.append(float(scores_np[i]))
        else:
            confidences = [None] * len(boxes_xyxy)
        
        # Create detections using helper
        detections = [
            self._create_detection(
                mask=selected_masks[i],
                box_xyxy=boxes_xyxy[i],
                label=labels[i],
                width=width,
                height=height,
                index=i,
                confidence=confidences[i]
            )
            for i in range(len(boxes_xyxy))
        ]
        
        return Detections(detections=detections)
    
    def _convert_fo_boxes_to_sam3(
        self,
        detections: Detections,
        width: int,
        height: int
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Convert FiftyOne Detections to SAM3 box format.
        
        Args:
            detections: fo.Detections with bounding boxes
            width: Image width
            height: Image height
            
        Returns:
            Tuple of (boxes, labels):
            - boxes: [[x1, y1, x2, y2], ...] in absolute xyxy format
            - labels: [1, 1, ...] (1=positive box, 0=negative)
        """
        boxes = []
        labels = []
        
        for det in detections.detections:
            # FiftyOne format: [rel_x, rel_y, rel_w, rel_h]
            rel_x, rel_y, rel_w, rel_h = det.bounding_box
            
            # Convert to absolute xyxy
            x1 = rel_x * width
            y1 = rel_y * height
            x2 = (rel_x + rel_w) * width
            y2 = (rel_y + rel_h) * height
            
            boxes.append([x1, y1, x2, y2])
            
            # Check if detection has "negative" attribute (default positive)
            label = 0 if getattr(det, 'negative', False) else 1
            labels.append(label)
        
        return boxes, labels
    
    def _masks_to_boxes(self, masks: torch.Tensor, mask_index: int = 0) -> np.ndarray:
        """
        Extract bounding boxes from masks.
        
        Args:
            masks: [N, num_masks, H, W] or [N, H, W]
            mask_index: Which mask to use if multiple
            
        Returns:
            [N, 4] array of boxes in xyxy format
        """
        masks_np = masks.cpu().numpy()
        
        # Select specific mask if multiple
        if masks_np.ndim == 4:
            masks_np = masks_np[:, mask_index]  # [N, H, W]
        
        boxes = []
        for mask in masks_np:
            # Find non-zero pixels
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                boxes.append([x1, y1, x2, y2])
            else:
                # Empty mask - use zeros
                boxes.append([0, 0, 1, 1])
        
        return np.array(boxes)