import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import Sam3Model

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the SAM3 model from HuggingFace.
    
    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load a SAM3 model for use with FiftyOne.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID or path to model files
            Default: "facebook/sam3"
        **kwargs: Additional parameters:
            - operation: Operation type (concept_segmentation, visual_segmentation, automatic_segmentation)
            - prompt: Default text prompt for concept_segmentation
            - threshold: Confidence threshold (default: 0.5)
            - mask_threshold: Mask binarization threshold (default: 0.5)
            - points_mask_index: Which mask to use for point prompts (0, 1, or 2)
            - auto_kwargs: Dict of kwargs for automatic mask generation
            - pooling_strategy: Embeddings pooling strategy (mean, max, cls)
            - device: Device to use (cuda, cpu, mps, or None for auto)
    
    Returns:
        Sam3Model: Initialized model ready for inference
    
    Example:
        # Concept segmentation with text
        model = load_model(
            model_path="facebook/sam3",
            operation="concept_segmentation",
            prompt="person",
            threshold=0.5
        )
        dataset.apply_model(model, label_field="people", batch_size=16)
        
        # Visual segmentation with boxes
        model = load_model(
            operation="visual_segmentation",
            points_mask_index=0
        )
        dataset.apply_model(
            model,
            label_field="masks",
            prompt_field="detections",
            batch_size=8
        )
        
        # Automatic segmentation
        model = load_model(
            operation="automatic_segmentation",
            auto_kwargs={"points_per_batch": 64}
        )
        dataset.apply_model(model, label_field="auto_masks", batch_size=4)
    """
    if model_path is None:
        model_path = "facebook/sam3"
    
    # Pass all parameters directly to Sam3Model
    return Sam3Model(model_path=model_path, **kwargs)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.
    
    Args:
        model_name: the name of the model
        ctx: an ExecutionContext
    
    Returns:
        a fiftyone.operators.types.Property
    """
    inputs = types.Object()
    
    # Operation selection
    inputs.enum(
        "operation",
        values=[
            "concept_segmentation",
            "visual_segmentation",
            "automatic_segmentation"
        ],
        default="concept_segmentation",
        label="Operation",
        description="Type of segmentation to perform",
    )
    
    # Text prompt for concept segmentation
    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Text Prompt",
        description="Default text prompt for concept segmentation (e.g., 'cat', 'person')",
    )
    
    # Segmentation thresholds
    inputs.float(
        "threshold",
        default=0.5,
        label="Confidence Threshold",
        description="Minimum confidence score for detections",
    )
    
    inputs.float(
        "mask_threshold",
        default=0.5,
        label="Mask Threshold",
        description="Threshold for mask binarization",
    )
    
    # Point prompt settings
    inputs.int(
        "points_mask_index",
        default=0,
        label="Points Mask Index",
        description="Which mask to use for point prompts (0=best quality, 1, 2)",
    )
    
    # Automatic segmentation settings
    inputs.int(
        "points_per_batch",
        default=64,
        label="Points Per Batch (Auto)",
        description="Points per batch for automatic segmentation",
    )
    
    # Embeddings settings
    inputs.enum(
        "pooling_strategy",
        values=["mean", "max", "cls"],
        default="mean",
        label="Embeddings Pooling",
        description="Pooling strategy for embeddings",
    )
    
    # Device selection
    inputs.enum(
        "device",
        values=["auto", "cuda", "cpu", "mps"],
        default="auto",
        label="Device",
        description="Device to use for inference",
    )
    
    return types.Property(inputs)

