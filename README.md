# Custom Vision Data Export Tool

Export your Azure Custom Vision projects, images, tags, and region annotations before the service is deprecated.

## Features

- Export all projects from your Custom Vision account
- Download all images with original quality
- Save metadata including tags, labels, and ROI annotations
- Convert to COCO or YOLO format for use with other ML frameworks
- Interactive browser to explore exported data

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your Custom Vision credentials as environment variables:

```bash
export CUSTOM_VISION_ENDPOINT="https://eastus.api.cognitive.microsoft.com/"
export CUSTOM_VISION_KEY="your-training-key"
```

## Usage

### Export All Data

```bash
python custom_vision_export.py
```

This will:
- List all your Custom Vision projects
- Download all images from each project
- Save metadata, tags, and region annotations
- Create `annotations.json` and `metadata.json` for each project

### Browse Exported Data

```bash
# Interactive browser
python browse_data.py

# Quick summary
python browse_data.py --summary
```

### Export Formats

The browser can convert your data to popular formats:

- **COCO format**: For use with detectron2, MMDetection, etc.
- **YOLO format**: For use with YOLOv5, YOLOv8, etc.

## Output Structure

```
custom_vision_export/
├── export_summary.json
├── Project1/
│   ├── metadata.json      # Full metadata including original URIs
│   ├── annotations.json   # Simplified annotations
│   └── images/           # Downloaded images
├── Project2/
│   └── ...
```

## Data Format

### annotations.json

```json
[
  {
    "image_id": "...",
    "filename": "image.jpg",
    "width": 1920,
    "height": 1080,
    "tags": ["cat", "animal"],
    "regions": [
      {
        "tag": "cat",
        "bbox": {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}
      }
    ]
  }
]
```
