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
    
    def __init__(
        self, 
        field_mapping=None, 
        operation=None,
        text_prompt_field=None,
        box_prompt_field=None,
        point_prompt_field=None
    ):
        self.operation = operation
        self.text_prompt_field = text_prompt_field
        self.box_prompt_field = box_prompt_field
        self.point_prompt_field = point_prompt_field
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        """Keys needed from each sample based on operation."""
        keys = ["filepath"]
        
        # Automatic segmentation needs no prompts
        if self.operation == "automatic_segmentation":
            return keys
        
        # Add prompt fields if specified
        if self.text_prompt_field:
            keys.append(self.text_prompt_field)
        if self.box_prompt_field:
            keys.append(self.box_prompt_field)
        if self.point_prompt_field:
            keys.append(self.point_prompt_field)
        
        return keys
    
    def __call__(self, sample_dict):
        """
        Load image and extract prompts. Runs in DataLoader workers.
        
        Returns:
            Dict with 'image', 'original_size', and prompt fields
        """
        # Load image
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        
        result = {
            'image': image,
            'original_size': image.size,  # (width, height)
            'text_prompt': None,
            'box_prompts': None,
            'point_prompts': None,
        }
        
        # Extract prompts for non-automatic operations
        if self.operation != "automatic_segmentation":
            if self.text_prompt_field and self.text_prompt_field in sample_dict:
                result['text_prompt'] = sample_dict[self.text_prompt_field]
            
            if self.box_prompt_field and self.box_prompt_field in sample_dict:
                result['box_prompts'] = sample_dict[self.box_prompt_field]
            
            if self.point_prompt_field and self.point_prompt_field in sample_dict:
                result['point_prompts'] = sample_dict[self.point_prompt_field]
        
        return result


class Sam3Model(Model, SupportsGetItem, TorchModelMixin):
    """
    Unified SAM3 model for FiftyOne with batching support.
    
    Example usage:
        # Concept segmentation with text
        model = Sam3Model(operation="concept_segmentation", prompt="cat")
        dataset.apply_model(model, label_field="cats", batch_size=16)
        
        # Visual segmentation with box prompts
        model = Sam3Model(operation="visual_segmentation")
        dataset.apply_model(
            model, 
            label_field="masks",
            prompt_field="detections",
            batch_size=8
        )
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
            Sam3TrackerModel, Sam3TrackerProcessor,
            pipeline
        )
        
        logger.info(f"Loading SAM3 for operation: {self._operation}")
        
        if self._operation == "concept_segmentation":
            # Concept segmentation: text/visual prompts → all matching instances
            self.model = Sam3Model.from_pretrained(self.model_path).to(self.device)
            self.processor = Sam3Processor.from_pretrained(self.model_path)
        
        else:  # visual_segmentation or automatic_segmentation
            # Visual segmentation: interactive prompts → specific instances
            self.model = Sam3TrackerModel.from_pretrained(self.model_path).to(self.device)
            self.processor = Sam3TrackerProcessor.from_pretrained(self.model_path)
            
            # Load pipeline for automatic segmentation
            if self._operation == "automatic_segmentation":
                device_id = 0 if self.device.type == "cuda" else -1
                self.mask_generator = pipeline(
                    "mask-generation",
                    model=self.model_path,
                    device=device_id
                )
        
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
        # Extract prompt field names from needs_fields
        text_prompt_field = self._fields.get("text_prompt_field")
        box_prompt_field = self._fields.get("box_prompt_field")
        point_prompt_field = self._fields.get("point_prompt_field")
        
        # Handle generic "prompt_field" - map to appropriate field based on operation
        if "prompt_field" in self._fields:
            prompt_field = self._fields["prompt_field"]
            
            if self._operation == "concept_segmentation":
                # Could be text or boxes - check both if not specified
                if not text_prompt_field:
                    text_prompt_field = prompt_field
                if not box_prompt_field:
                    box_prompt_field = prompt_field
            
            elif self._operation == "visual_segmentation":
                # Could be boxes or points - check both if not specified
                if not box_prompt_field:
                    box_prompt_field = prompt_field
                if not point_prompt_field:
                    point_prompt_field = prompt_field
        
        return Sam3GetItem(
            field_mapping=field_mapping,
            operation=self._operation,
            text_prompt_field=text_prompt_field,
            box_prompt_field=box_prompt_field,
            point_prompt_field=point_prompt_field
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
        
        # Extract prompts from sample if available
        text_prompt = self.default_prompt
        box_prompts = None
        point_prompts = None
        
        if sample is not None:
            # Try to get prompt from configured field
            for field_key in ["prompt_field", "text_prompt_field", "box_prompt_field", "point_prompt_field"]:
                if field_key in self._fields:
                    field_name = self._fields[field_key]
                    if sample.has_field(field_name):
                        field_value = sample.get_field(field_name)
                        
                        # Assign to appropriate prompt type
                        if isinstance(field_value, str):
                            text_prompt = field_value
                        elif isinstance(field_value, Detections):
                            box_prompts = field_value
                        elif isinstance(field_value, Keypoints):
                            point_prompts = field_value
                        
                        break  # Use first found field
        
        # Create batch item
        batch_item = {
            'image': image,
            'original_size': image.size,
            'text_prompt': text_prompt,
            'box_prompts': box_prompts,
            'point_prompts': point_prompts,
        }
        
        # Use batch inference
        results = self.predict_all([batch_item])
        return results[0]
    
    def predict_all(self, batch, preprocess=None):
        """
        Batch inference - routes to operation-specific method.
        
        Args:
            batch: List of dicts from GetItem
            preprocess: Whether to apply preprocessing (unused, handled by GetItem)
            
        Returns:
            List of fo.Detections (one per image)
        """
        if not batch:
            return []
        
        # Route to operation-specific method
        if self._operation == "concept_segmentation":
            return self._predict_concept_segmentation(batch)
        elif self._operation == "visual_segmentation":
            return self._predict_visual_segmentation(batch)
        elif self._operation == "automatic_segmentation":
            return self._predict_automatic_segmentation(batch)
        else:
            raise ValueError(f"Unknown operation: {self._operation}")
    
    # ==================== OPERATION IMPLEMENTATIONS ====================
    
    def _predict_concept_segmentation(self, batch):
        """
        Concept segmentation: Find ALL matching instances.
        Supports text prompts, box prompts, or both combined.
        """
        # Extract data from batch
        images = [item['image'] for item in batch]
        text_prompts = [item.get('text_prompt') or self.default_prompt for item in batch]
        box_prompts = [item.get('box_prompts') for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Prepare processor inputs
        processor_kwargs = {"images": images, "return_tensors": "pt"}
        
        # Add text prompts if any exist
        if any(text_prompts):
            processor_kwargs["text"] = text_prompts
        
        # Add box prompts if any exist
        if any(box_prompts):
            all_boxes = []
            all_box_labels = []
            
            for bp, (w, h) in zip(box_prompts, original_sizes):
                if bp and len(bp.detections) > 0:
                    boxes, labels = self._convert_fo_boxes_to_sam3(bp, w, h)
                    all_boxes.append(boxes)
                    all_box_labels.append(labels)
                else:
                    all_boxes.append(None)
                    all_box_labels.append(None)
            
            processor_kwargs["input_boxes"] = all_boxes
            processor_kwargs["input_boxes_labels"] = all_box_labels
        
        # Run SAM3 inference
        inputs = self.processor(**processor_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process to get instance segmentation
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        
        # Convert each result to FiftyOne format
        fo_results = []
        for i, result in enumerate(results):
            # Use text prompt as label, fallback to "object"
            label = text_prompts[i] if text_prompts[i] else "object"
            
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
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_visual_segmentation(self, batch):
        """
        Visual segmentation: Segment SPECIFIC instances.
        Supports point prompts or box prompts (detects type automatically).
        """
        box_prompts = [item.get('box_prompts') for item in batch]
        point_prompts = [item.get('point_prompts') for item in batch]
        
        # Determine which prompt type is present
        has_boxes = any(box_prompts)
        has_points = any(point_prompts)
        
        if has_boxes:
            return self._predict_visual_boxes(batch)
        elif has_points:
            return self._predict_visual_points(batch)
        else:
            # No prompts - return empty detections
            return [Detections() for _ in batch]
    
    def _predict_visual_boxes(self, batch):
        """Process visual segmentation with box prompts."""
        images = [item['image'] for item in batch]
        all_box_prompts = [item.get('box_prompts') for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Convert FO boxes to SAM3 format: [batch][num_objects][4]
        all_boxes = []
        all_labels_list = []
        
        for prompts, (w, h) in zip(all_box_prompts, original_sizes):
            if prompts and len(prompts.detections) > 0:
                boxes_for_image = []
                labels_for_image = []
                
                for det in prompts.detections:
                    # Convert relative [x, y, w, h] to absolute [x1, y1, x2, y2]
                    rel_x, rel_y, rel_w, rel_h = det.bounding_box
                    x1 = rel_x * w
                    y1 = rel_y * h
                    x2 = (rel_x + rel_w) * w
                    y2 = (rel_y + rel_h) * h
                    
                    boxes_for_image.append([x1, y1, x2, y2])
                    labels_for_image.append(det.label)
                
                all_boxes.append([boxes_for_image])  # Extra list for SAM3 format
                all_labels_list.append(labels_for_image)
            else:
                # Empty prompts - use dummy box
                all_boxes.append([[[0, 0, 100, 100]]])
                all_labels_list.append(["object"])
        
        # Process with SAM3 Tracker
        inputs = self.processor(
            images=images,
            input_boxes=all_boxes,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Post-process masks
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )
        
        # Convert to FiftyOne Detections
        fo_results = []
        for i, (mask, labels, prompts, (w, h)) in enumerate(
            zip(masks, all_labels_list, all_box_prompts, original_sizes)
        ):
            if prompts and len(prompts.detections) > 0:
                # Get boxes in xyxy format
                boxes_xyxy = np.array([
                    [det.bounding_box[0] * w,
                     det.bounding_box[1] * h,
                     (det.bounding_box[0] + det.bounding_box[2]) * w,
                     (det.bounding_box[1] + det.bounding_box[3]) * h]
                    for det in prompts.detections
                ])
                
                detections = self._tracker_to_detections(
                    masks=mask,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    width=w,
                    height=h
                )
            else:
                detections = Detections()
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_visual_points(self, batch):
        """Process visual segmentation with point prompts."""
        images = [item['image'] for item in batch]
        all_point_prompts = [item.get('point_prompts') for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        # Convert FO keypoints to SAM3 format
        all_points = []
        all_labels = []
        all_label_names = []
        
        for prompts, (w, h) in zip(all_point_prompts, original_sizes):
            if prompts and len(prompts.keypoints) > 0:
                points_for_image = []
                labels_for_image = []
                names_for_image = []
                
                for kp in prompts.keypoints:
                    kp_points = []
                    kp_labels = []
                    
                    for point in kp.points:
                        # Convert relative to absolute
                        rel_x, rel_y = point
                        abs_x = rel_x * w
                        abs_y = rel_y * h
                        kp_points.append([abs_x, abs_y])
                        kp_labels.append(1)  # Positive point
                    
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
            outputs = self.model(**inputs, multimask_output=True)
        
        # Post-process - select best mask using points_mask_index
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )
        
        # Convert to FiftyOne Detections
        fo_results = []
        for i, (mask, names, prompts, (w, h)) in enumerate(
            zip(masks, all_label_names, all_point_prompts, original_sizes)
        ):
            if prompts and len(prompts.keypoints) > 0:
                # Extract boxes from masks
                boxes_xyxy = self._masks_to_boxes(mask, self.points_mask_index)
                
                detections = self._tracker_to_detections(
                    masks=mask,
                    boxes_xyxy=boxes_xyxy,
                    labels=names,
                    width=w,
                    height=h,
                    mask_index=self.points_mask_index
                )
            else:
                detections = Detections()
            
            fo_results.append(detections)
        
        return fo_results
    
    def _predict_automatic_segmentation(self, batch):
        """
        Automatic segmentation: Generate all masks without prompts.
        Uses HuggingFace mask generation pipeline.
        """
        fo_results = []
        
        for item in batch:
            image = item['image']
            w, h = item['original_size']
            
            # Run mask generation pipeline
            output = self.mask_generator(
                image,
                points_per_batch=self.auto_kwargs.get("points_per_batch", 64)
            )
            
            # Convert pipeline output to FiftyOne
            if output and "masks" in output and len(output["masks"]) > 0:
                detections = self._pipeline_to_detections(output, w, h)
            else:
                detections = Detections()
            
            fo_results.append(detections)
        
        return fo_results
    
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
        # Load image
        if isinstance(item, dict):
            # Item from GetItem DataLoader - already has PIL Image
            image = item['image']
        elif isinstance(item, str):
            image = Image.open(item).convert("RGB")
        elif hasattr(item, 'filepath'):
            image = Image.open(item.filepath).convert("RGB")
        elif hasattr(item, 'path'):
            image = Image.open(item.path).convert("RGB")
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
            fo.Detections
        """
        detections = []
        
        # Convert to numpy
        masks_np = masks.cpu().numpy()
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        for i in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[i]
            
            # Convert absolute xyxy to relative [x, y, width, height]
            rel_x = x1 / width
            rel_y = y1 / height
            rel_w = (x2 - x1) / width
            rel_h = (y2 - y1) / height
            
            # Crop mask to bounding box
            mask = masks_np[i]
            y1_int, y2_int = int(round(y1)), int(round(y2))
            x1_int, x2_int = int(round(x1)), int(round(x2))
            cropped_mask = mask[y1_int:y2_int, x1_int:x2_int]
            
            detection = Detection(
                label=label,
                bounding_box=[rel_x, rel_y, rel_w, rel_h],
                mask=cropped_mask,
                confidence=float(scores_np[i])
            )
            detections.append(detection)
        
        return Detections(detections=detections)
    
    def _tracker_to_detections(
        self,
        masks: torch.Tensor,
        boxes_xyxy: np.ndarray,
        labels: List[str],
        width: int,
        height: int,
        mask_index: int = 0
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
            
        Returns:
            fo.Detections
        """
        detections = []
        
        masks_np = masks.cpu().numpy()
        
        for i, (box, label) in enumerate(zip(boxes_xyxy, labels)):
            x1, y1, x2, y2 = box
            
            # Convert to relative coordinates
            rel_x = x1 / width
            rel_y = y1 / height
            rel_w = (x2 - x1) / width
            rel_h = (y2 - y1) / height
            
            # Select mask - handle both [N, num_masks, H, W] and [N, H, W]
            if masks_np.ndim == 4:
                mask = masks_np[i, mask_index]  # Select specific mask
            else:
                mask = masks_np[i]
            
            # Crop mask to bounding box
            y1_int, y2_int = int(round(y1)), int(round(y2))
            x1_int, x2_int = int(round(x1)), int(round(x2))
            cropped_mask = mask[y1_int:y2_int, x1_int:x2_int]
            
            detection = Detection(
                label=label,
                bounding_box=[rel_x, rel_y, rel_w, rel_h],
                mask=cropped_mask
            )
            detections.append(detection)
        
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
    
    def _pipeline_to_detections(
        self,
        output: Dict,
        width: int,
        height: int
    ) -> Detections:
        """
        Convert mask generation pipeline output to FiftyOne Detections.
        
        Args:
            output: Pipeline output dict with "masks" key
            width: Image width
            height: Image height
            
        Returns:
            fo.Detections
        """
        detections = []
        
        masks = output["masks"]
        
        for i, mask in enumerate(masks):
            # Convert mask to numpy if needed
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Extract bounding box from mask
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                
                # Convert to relative coordinates
                rel_x = x1 / width
                rel_y = y1 / height
                rel_w = (x2 - x1) / width
                rel_h = (y2 - y1) / height
                
                # Crop mask to bounding box
                cropped_mask = mask_np[y1:y2+1, x1:x2+1]
                
                detection = Detection(
                    label=f"object_{i}",
                    bounding_box=[rel_x, rel_y, rel_w, rel_h],
                    mask=cropped_mask
                )
                detections.append(detection)
        
        return Detections(detections=detections)

