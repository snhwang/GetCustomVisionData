# Custom Vision Data Export Tool

Export your Azure Custom Vision projects, images, tags, and region annotations before the service is deprecated.

## Features

- Export all projects from your Custom Vision account
- Download all images with original quality
- Save metadata including tags, labels, and ROI annotations
- Resume interrupted exports (skips already downloaded images)
- Convert to COCO or YOLO format for use with other ML frameworks
- Web-based viewer to browse images, tags, and ROI annotations
- REST API for programmatic access

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your Custom Vision credentials:

```
ENDPOINT=https://eastus.api.cognitive.microsoft.com/
CUSTOM_VISION_KEY=your-training-key
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
- Skip already downloaded images (safe to re-run after interruptions)

### Web Viewer

Browse your exported data with the web-based viewer:

```bash
python app/main.py
```

Then open http://localhost:8200 in your browser.

**Viewer features:**
- Project selector dropdown
- Image thumbnail list with tag preview
- Filter images by tag
- Search by filename or tag name
- Full image display with ROI bounding box overlays
- Toggle ROI visibility
- Metadata panel showing dimensions, tags, and regions
- Keyboard navigation (arrow keys or A/D)

### Command Line Browser

```bash
# Interactive browser
python browse_data.py

# Quick summary
python browse_data.py --summary
```

### REST API

The web viewer also exposes a REST API:

| Endpoint | Description |
|----------|-------------|
| `GET /projects` | List all exported projects |
| `GET /projects/{name}` | Get project details |
| `GET /projects/{name}/tags` | Get project tags |
| `GET /projects/{name}/images` | Get images (supports filtering) |
| `GET /projects/{name}/images/{id}/file` | Download image file |
| `GET /projects/{name}/export/coco` | Export to COCO format |
| `GET /projects/{name}/export/yolo` | Export to YOLO format |
| `GET /health` | Health check |

API documentation available at http://localhost:8200/docs

### Export Formats

Convert your data to popular formats:

- **COCO format**: For use with detectron2, MMDetection, etc.
- **YOLO format**: For use with YOLOv5, YOLOv8, etc.

## Output Structure

```
custom_vision_export/
├── export_summary.json
├── Project1/
│   ├── metadata.json      # Full metadata including original URIs
│   ├── annotations.json   # Simplified annotations
│   └── images/            # Downloaded images
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

## Multiple Accounts

To export from multiple Custom Vision accounts, update the `.env` file with different credentials and run the export again. Projects are merged into the same output directory without overwriting existing data.
