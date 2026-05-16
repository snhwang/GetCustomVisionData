"""
Custom Vision Data API

FastAPI web application for browsing and exporting Custom Vision data.
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from browse_data import DataBrowser
from custom_vision_export import CustomVisionExporter
import config

app = FastAPI(
    title="Custom Vision Data API",
    description="API for browsing and exporting Azure Custom Vision data",
    version="1.0.0"
)

browser = DataBrowser(config.OUTPUT_DIR)

# Path to templates
TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the viewer UI."""
    viewer_path = TEMPLATES_DIR / "viewer.html"
    if viewer_path.exists():
        return viewer_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Viewer not found</h1>", status_code=404)


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "message": "Custom Vision Data API",
        "endpoints": {
            "viewer": "/",
            "projects": "/projects",
            "tags": "/projects/{project_name}/tags",
            "images": "/projects/{project_name}/images",
            "export": "/export",
            "docs": "/docs"
        }
    }


@app.get("/projects")
async def list_projects():
    """List all exported projects."""
    projects = browser.list_projects()
    return {"projects": projects, "count": len(projects)}


@app.get("/projects/{project_name}")
async def get_project(project_name: str):
    """Get details for a specific project."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    data = browser.load_project(project["path"])
    return {
        "project": data["project"],
        "summary": data["summary"],
        "tags": data["tags"]
    }


@app.get("/projects/{project_name}/tags")
async def get_project_tags(project_name: str):
    """Get all tags for a project."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    tags = browser.get_tags(project["path"])
    return {"tags": tags, "count": len(tags)}


@app.get("/projects/{project_name}/images")
async def get_project_images(
    project_name: str,
    tag: Optional[str] = Query(None, description="Filter by tag name"),
    with_regions: bool = Query(False, description="Only images with region annotations"),
    search: Optional[str] = Query(None, description="Search images by tag"),
    skip: int = Query(0, ge=0, description="Number of images to skip"),
    limit: int = Query(100, ge=1, description="Maximum images to return")
):
    """Get images from a project with optional filtering."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    if tag:
        images = browser.get_images_by_tag(project["path"], tag)
    elif with_regions:
        images = browser.get_images_with_regions(project["path"])
    elif search:
        images = browser.search_images(project["path"], search)
    else:
        data = browser.load_project(project["path"])
        images = data["images"]

    total = len(images)
    images = images[skip:skip + limit]

    return {
        "images": images,
        "total": total,
        "skip": skip,
        "limit": limit,
        "returned": len(images)
    }


@app.get("/projects/{project_name}/images/{image_id}")
async def get_image_metadata(project_name: str, image_id: str):
    """Get metadata for a specific image."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    data = browser.load_project(project["path"])
    image = next((img for img in data["images"] if img["id"] == image_id), None)

    if not image:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")

    return image


@app.get("/projects/{project_name}/images/{image_id}/file")
async def get_image_file(project_name: str, image_id: str):
    """Download the actual image file."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    data = browser.load_project(project["path"])
    image = next((img for img in data["images"] if img["id"] == image_id), None)

    if not image:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")

    image_path = browser.get_image_path(project["path"], image)

    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(image_path)


@app.get("/projects/{project_name}/export/coco")
async def export_coco(project_name: str):
    """Export project annotations in COCO format."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    coco_data = browser.export_coco_format(project["path"])
    return coco_data


@app.get("/projects/{project_name}/export/yolo")
async def export_yolo(project_name: str):
    """Export project annotations in YOLO format (returns class mapping and sample)."""
    projects = browser.list_projects()
    project = next((p for p in projects if p["name"] == project_name), None)

    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    data = browser.load_project(project["path"])
    class_names = [tag["name"] for tag in data["tags"]]

    # Generate YOLO format annotations in memory
    annotations = []
    for img in data["images"]:
        if not img.get("local_filename") or not img["regions"]:
            continue

        img_annotations = []
        for region in img["regions"]:
            class_id = class_names.index(region["tag_name"]) if region["tag_name"] in class_names else 0
            center_x = region["left"] + region["width"] / 2
            center_y = region["top"] + region["height"] / 2
            img_annotations.append({
                "class_id": class_id,
                "class_name": region["tag_name"],
                "center_x": center_x,
                "center_y": center_y,
                "width": region["width"],
                "height": region["height"]
            })

        if img_annotations:
            annotations.append({
                "filename": img["local_filename"],
                "annotations": img_annotations
            })

    return {
        "classes": class_names,
        "annotations": annotations,
        "total_images": len(annotations)
    }


@app.post("/export/run")
async def run_export(download_images: bool = True):
    """Trigger a fresh export from Custom Vision API."""
    if not config.TRAINING_KEY:
        raise HTTPException(
            status_code=500,
            detail="CUSTOM_VISION_KEY not configured. Set it in .env file."
        )

    try:
        exporter = CustomVisionExporter(
            endpoint=config.ENDPOINT,
            training_key=config.TRAINING_KEY,
            output_dir=config.OUTPUT_DIR
        )

        projects = exporter.list_projects()

        return {
            "message": f"Found {len(projects)} project(s). Export started.",
            "projects": [{"name": p.name, "id": str(p.id)} for p in projects],
            "note": "Use GET /projects to see exported data after completion."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "config": {
            "endpoint_configured": bool(config.ENDPOINT),
            "key_configured": bool(config.TRAINING_KEY),
            "output_dir": config.OUTPUT_DIR
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8200, reload=True)
