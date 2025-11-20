# SAM3 for FiftyOne

![image](sam3_images.gif)

Integration of Meta's [SAM3 (Segment Anything Model 3)](https://huggingface.co/facebook/sam3) into FiftyOne, with full batching support and visual embeddings.

## Features

✅ **Three Segmentation Operations**

- **Concept Segmentation**: Find ALL matching instances using text prompts

- **Visual Segmentation**: Segment SPECIFIC instances using interactive prompts (boxes/points)

- **Automatic Segmentation**: Generate all masks without prompts (with quality filtering & deduplication)

✅ **Visual Embeddings**
- Extract 1024-dim visual embeddings for similarity search
- Three pooling strategies: mean, max, cls
- Independent of text prompts

## Installation

**⚠️ Important:** SAM3 is brand new and requires transformers from source (not yet on PyPI):

```bash
# Install transformers from source
pip install git+https://github.com/huggingface/transformers.git#egg=transformers

# Install FiftyOne
pip install fiftyone

# Install other dependencies
pip install torch torchvision huggingface-hub
```

## Loading as Remote Zoo Model

```python
import fiftyone.zoo as foz

# Register the remote model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/sam3_images"
)

# Load the model
model = foz.load_zoo_model("facebook/sam3")
```

## Parameters

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | str | "concept_segmentation" | Operation type |
| `prompt` | str | None | Default text prompt |
| `threshold` | float | 0.5 | Confidence threshold |
| `mask_threshold` | float | 0.5 | Mask binarization threshold |
| `points_mask_index` | int | 0 | Which mask for point prompts (0=best) |
| `auto_kwargs` | dict | `{}` | Automatic segmentation settings |
| `auto_kwargs.points_per_side` | int | 16 | Point grid density (16² = 256 points) |
| `auto_kwargs.points_per_batch` | int | 256 | Inference batch size |
| `auto_kwargs.quality_threshold` | float | 0.8 | Minimum quality score to keep (0-1) |
| `auto_kwargs.iou_threshold` | float | 0.85 | IoU threshold for deduplication (0-1) |
| `auto_kwargs.max_masks` | int | None | Maximum masks to return (None = unlimited) |
| `pooling_strategy` | str | "mean" | Embeddings pooling (mean/max/cls) |
| `return_semantic_seg` | bool | False | Include semantic segmentation mask |
| `device` | str | "auto" | Device (cuda/cpu/mps/auto) |

### Operations

**concept_segmentation**
- Finds ALL matching instances
- Supports text prompts only (single string or list)
- Returns all objects matching the concept

**visual_segmentation**
- Segments SPECIFIC instances
- Supports box or point prompts
- Returns one mask per prompt

**automatic_segmentation**
- No prompts needed
- Generates all possible masks
- Memory intensive

## Complete Example (All Operations)

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Load dataset
dataset = foz.load_zoo_dataset("quickstart")

# Register remote zoo model
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/sam3_images"
)

# Load SAM3 model
model = foz.load_zoo_model("facebook/sam3")

# ============================================================
# 1. Compute Embeddings for Similarity Search
# ============================================================
model.pooling_strategy = "max"  # or "mean", "cls"

dataset.compute_embeddings(
    model,
    embeddings_field="sam_embeddings",
    batch_size=32
)

# Visualize with UMAP
fob.compute_visualization(
    dataset,
    method="umap",
    brain_key="sam_viz",
    embeddings="sam_embeddings",
    num_dims=2
)

# ============================================================
# 2. Automatic Segmentation (Segment Everything)
# ============================================================
model.operation = "automatic_segmentation"
model.threshold = 0.5
model.mask_threshold = 0.5

dataset.apply_model(
    model,
    label_field="automatic_segmentation",
    batch_size=4,
    num_workers=4
)

# ============================================================
# 3. Visual Segmentation (Refine Existing Detections)
# ============================================================
model.operation = "visual_segmentation"

dataset.apply_model(
    model,
    label_field="visual_segmentation",
    prompt_field="ground_truth",  # Use existing detections as prompts
    batch_size=64,
    num_workers=4
)

# ============================================================
# 4. Concept Segmentation (Find Multiple Object Types)
# ============================================================
model.operation = "concept_segmentation"
model.prompt = [
    "bird", "human", "land vehicle", "air vehicle",
    "aquatic vehicle", "animal", "food", "utensils", "furniture"
]
model.threshold = 0.5
model.mask_threshold = 0.5

dataset.apply_model(
    model,
    label_field="concept_segmentation",
    batch_size=8,
    num_workers=4
)

# Launch app
session = fo.launch_app(dataset)
```

## Quick Start

### Concept Segmentation (Text Prompts)

Find ALL instances matching text concepts:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = fo.load_dataset("quickstart")

# Load model
model = foz.load_zoo_model("facebook/sam3")

# Single concept
model.operation = "concept_segmentation"
model.prompt = "person"
model.threshold = 0.5
model.mask_threshold = 0.5

dataset.apply_model(
    model,
    label_field="people",
    batch_size=16,
    num_workers=4
)

# Multiple concepts (finds all in each image)
# Note: Runs one inference pass per concept, so 4 concepts = 4x slower
model.prompt = ["person", "car", "dog", "bird"]
model.threshold = 0.5
model.mask_threshold = 0.5

dataset.apply_model(
    model,
    label_field="multiple_objects",
    batch_size=8,
    num_workers=4
)

session = fo.launch_app(dataset)
```

### Visual Segmentation (Box or Point Prompts)

Segment SPECIFIC instances using existing detections or keypoints as prompts:

```python
# Load model
model = foz.load_zoo_model("facebook/sam3")

# Configure for visual segmentation
model.operation = "visual_segmentation"

# Option 1: Use boxes as prompts
dataset.apply_model(
    model,
    label_field="box_segmentations",
    prompt_field="ground_truth",  # Field with fo.Detections (boxes)
    batch_size=64,
    num_workers=4
)

# Option 2: Use keypoints as prompts
dataset.apply_model(
    model,
    label_field="point_segmentations",
    prompt_field="keypoints",  # Field with fo.Keypoints (points)
    batch_size=64,
    num_workers=4
)
```

### Automatic Segmentation

Generate all masks without prompts using point grid sampling with automatic filtering:

```python
# Load model
model = foz.load_zoo_model("facebook/sam3")

# Configure for automatic segmentation
model.operation = "automatic_segmentation"
model.auto_kwargs = {
    "points_per_side": 16,       # Grid density (16x16 = 256 points)
    "points_per_batch": 256,     # Inference batch size
    "quality_threshold": 0.8,    # Keep masks with IoU score >= 0.8
    "iou_threshold": 0.85,       # Remove duplicates with IoU > 0.85
    "max_masks": 100             # Limit to top 100 masks by quality
}

dataset.apply_model(
    model,
    label_field="auto_masks",
    batch_size=4,
    num_workers=2
)
```

**Quality of Life Features:**
- **Quality filtering**: Only keeps high-quality masks (configurable threshold)
- **Deduplication**: Removes overlapping/duplicate masks using NMS
- **Limit results**: Optionally cap at top N masks by quality score
- **Clear labeling**: Masks labeled as `object_0`, `object_1`, etc. for easy visualization

### Visual Embeddings

Extract embeddings for similarity search and visualization:

```python
import fiftyone.brain as fob

# Load model
model = foz.load_zoo_model("facebook/sam3")

# Configure pooling strategy
model.pooling_strategy = "max"  # or "mean", "cls"

# Compute embeddings
dataset.compute_embeddings(
    model,
    embeddings_field="sam3_embeddings",
    batch_size=32
)

# Similarity search
fob.compute_similarity(dataset, embeddings="sam3_embeddings")
query = dataset.first()
similar = dataset.sort_by_similarity(query, k=10)

# Visualize with UMAP
fob.compute_visualization(
    dataset,
    embeddings="sam3_embeddings",
    method="umap",
    num_dims=2
)
```

### Semantic Segmentation

SAM3 provides semantic segmentation alongside instance masks - a unified mask covering ALL instances:

```python
# Load model
model = foz.load_zoo_model("facebook/sam3")

# Configure
model.operation = "concept_segmentation"
model.prompt = "person"
model.return_semantic_seg = True  # Enable semantic segmentation

dataset.apply_model(
    model,
    label_field="instance_masks",
    batch_size=16,
    num_workers=4
)
```

### Per-Sample Text Prompts

```python
# Load model
model = foz.load_zoo_model("facebook/sam3")

# Configure
model.operation = "concept_segmentation"

# Assumes dataset has field "my_prompt" with text values
dataset.apply_model(
    model,
    label_field="results",
    prompt_field="my_prompt",  # Field with str or list values
    batch_size=16,
    num_workers=4
)

# Examples of valid prompts in dataset field:
# - "cat" (single string - finds cats in that image)
# - ["cat", "dog", "bird"] (list - finds all three in that image, 3x slower)
```

### Batching Limitations

**Visual Segmentation**: SAM3 Tracker cannot batch images with different numbers of prompts (boxes/points). The model automatically falls back to sequential processing when this occurs.

```python
# This will batch efficiently (all images have 1 box)
dataset.apply_model(model, prompt_field="single_box", batch_size=16)

# This will fall back to sequential (images have varying box counts)
# Performance will be slower but results are still correct
dataset.apply_model(model, prompt_field="multi_box", batch_size=16)
```

**Concept Segmentation**: No batching limitations - handles variable prompts naturally.

## Model Details

- **Model**: facebook/sam3
- **Embeddings**: 1024-dimensional visual features
- **Input**: RGB images (any size, resized to 1008x1008)
- **Output**: Instance masks with bounding boxes and scores

## Citation

```bibtex
@article{sam3,
  title={SAM 3: Segment Anything Model 3},
  author={Meta AI Research},
  year={2024}
}
```

## Links

- [Model Card](https://huggingface.co/facebook/sam3)
- [FiftyOne Docs](https://docs.voxel51.com/)
- [GitHub Issues](https://github.com/harpreetsahota204/sam3_images/issues)