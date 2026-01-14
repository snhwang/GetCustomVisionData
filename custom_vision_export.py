"""
Custom Vision Data Export Tool

Downloads all projects, images, tags, and region annotations from Azure Custom Vision
before the service is deprecated.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageRegion
from msrest.authentication import ApiKeyCredentials

import config


class CustomVisionExporter:
    """Export data from Azure Custom Vision projects."""

    def __init__(self, endpoint: str, training_key: str, output_dir: str = "custom_vision_export"):
        """Initialize the exporter with API credentials."""
        self.endpoint = endpoint
        self.training_key = training_key
        self.output_dir = Path(output_dir)

        # Create training client
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        self.trainer = CustomVisionTrainingClient(endpoint, credentials)

    def list_projects(self) -> list:
        """List all Custom Vision projects."""
        projects = self.trainer.get_projects()
        return projects

    def get_project_details(self, project_id: str) -> dict:
        """Get detailed information about a project."""
        project = self.trainer.get_project(project_id)
        return {
            "id": str(project.id),
            "name": project.name,
            "description": project.description,
            "project_type": project.settings.classification_type if hasattr(project.settings, 'classification_type') else None,
            "domain_id": str(project.settings.domain_id) if project.settings.domain_id else None,
            "target_export_platforms": project.settings.target_export_platforms if hasattr(project.settings, 'target_export_platforms') else [],
            "created": project.created.isoformat() if project.created else None,
            "last_modified": project.last_modified.isoformat() if project.last_modified else None,
        }

    def get_tags(self, project_id: str) -> list:
        """Get all tags for a project."""
        tags = self.trainer.get_tags(project_id)
        return [{
            "id": str(tag.id),
            "name": tag.name,
            "description": tag.description,
            "type": tag.type,
            "image_count": tag.image_count
        } for tag in tags]

    def get_all_images(self, project_id: str, tagged: Optional[bool] = None) -> list:
        """Get all images from a project with pagination."""
        all_images = []
        skip = 0
        take = 256  # Max allowed by API

        while True:
            if tagged is None:
                images = self.trainer.get_images(project_id, skip=skip, take=take)
            elif tagged:
                images = self.trainer.get_tagged_images(project_id, skip=skip, take=take)
            else:
                images = self.trainer.get_untagged_images(project_id, skip=skip, take=take)

            if not images:
                break

            all_images.extend(images)
            skip += take

            if len(images) < take:
                break

        return all_images

    def download_image(self, url: str, filepath: Path) -> bool:
        """Download an image from URL to filepath."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(response.content)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

    def export_image_metadata(self, image, tags_lookup: dict) -> dict:
        """Extract metadata from an image object."""
        metadata = {
            "id": str(image.id),
            "created": image.created.isoformat() if image.created else None,
            "width": image.width,
            "height": image.height,
            "original_image_uri": image.original_image_uri,
            "resized_image_uri": image.resized_image_uri,
            "thumbnail_uri": image.thumbnail_uri,
            "tags": [],
            "regions": []
        }

        # Add tag information
        if image.tags:
            for tag in image.tags:
                tag_info = {
                    "tag_id": str(tag.tag_id),
                    "tag_name": tags_lookup.get(str(tag.tag_id), "Unknown"),
                    "created": tag.created.isoformat() if tag.created else None
                }
                metadata["tags"].append(tag_info)

        # Add region/ROI information (for object detection projects)
        if image.regions:
            for region in image.regions:
                region_info = {
                    "region_id": str(region.region_id) if hasattr(region, 'region_id') else None,
                    "tag_id": str(region.tag_id),
                    "tag_name": tags_lookup.get(str(region.tag_id), "Unknown"),
                    "left": region.left,
                    "top": region.top,
                    "width": region.width,
                    "height": region.height,
                    "created": region.created.isoformat() if hasattr(region, 'created') and region.created else None
                }
                metadata["regions"].append(region_info)

        return metadata

    def export_project(self, project, download_images: bool = True) -> dict:
        """Export all data from a single project."""
        project_id = str(project.id)
        project_name = project.name

        # Create project directory
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in project_name)
        project_dir = self.output_dir / safe_name
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting project: {project_name}")

        # Get project details
        project_details = self.get_project_details(project_id)

        # Get all tags
        tags = self.get_tags(project_id)
        tags_lookup = {tag["id"]: tag["name"] for tag in tags}

        # Get all images
        print("  Fetching image list...")
        all_images = self.get_all_images(project_id)
        print(f"  Found {len(all_images)} images")

        # Process images
        images_dir = project_dir / "images"
        images_metadata = []

        for image in tqdm(all_images, desc="  Processing images"):
            metadata = self.export_image_metadata(image, tags_lookup)

            if download_images and image.original_image_uri:
                # Determine file extension from URL or default to jpg
                ext = ".jpg"
                if image.original_image_uri:
                    url_path = image.original_image_uri.split('?')[0]
                    if '.' in url_path.split('/')[-1]:
                        ext = '.' + url_path.split('.')[-1].lower()

                image_filename = f"{image.id}{ext}"
                image_path = images_dir / image_filename

                if self.download_image(image.original_image_uri, image_path):
                    metadata["local_filename"] = image_filename
                else:
                    metadata["local_filename"] = None
                    metadata["download_error"] = True

            images_metadata.append(metadata)

        # Compile export data
        export_data = {
            "project": project_details,
            "tags": tags,
            "images": images_metadata,
            "summary": {
                "total_images": len(images_metadata),
                "total_tags": len(tags),
                "images_with_tags": sum(1 for img in images_metadata if img["tags"]),
                "images_with_regions": sum(1 for img in images_metadata if img["regions"]),
            }
        }

        # Save metadata to JSON
        metadata_file = project_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved metadata to {metadata_file}")

        # Also save a simplified annotations file (useful for training)
        annotations = []
        for img in images_metadata:
            ann = {
                "image_id": img["id"],
                "filename": img.get("local_filename"),
                "width": img["width"],
                "height": img["height"],
                "tags": [t["tag_name"] for t in img["tags"]],
                "regions": [{
                    "tag": r["tag_name"],
                    "bbox": {
                        "left": r["left"],
                        "top": r["top"],
                        "width": r["width"],
                        "height": r["height"]
                    }
                } for r in img["regions"]]
            }
            annotations.append(ann)

        annotations_file = project_dir / "annotations.json"
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"  Saved annotations to {annotations_file}")

        return export_data

    def export_all(self, download_images: bool = True) -> dict:
        """Export all projects and their data."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Fetching projects from Custom Vision...")
        projects = self.list_projects()
        print(f"Found {len(projects)} project(s)")

        all_exports = {}

        for project in projects:
            export_data = self.export_project(project, download_images)
            all_exports[project.name] = export_data

        # Save summary
        summary = {
            "total_projects": len(projects),
            "projects": [{
                "name": p.name,
                "id": str(p.id),
                "image_count": all_exports[p.name]["summary"]["total_images"]
            } for p in projects]
        }

        summary_file = self.output_dir / "export_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nExport complete! Summary saved to {summary_file}")
        return all_exports


def main():
    """Main entry point."""
    exporter = CustomVisionExporter(
        endpoint=config.ENDPOINT,
        training_key=config.TRAINING_KEY,
        output_dir=config.OUTPUT_DIR
    )

    # List projects first
    print("=" * 60)
    print("Custom Vision Data Export Tool")
    print("=" * 60)

    try:
        projects = exporter.list_projects()

        if not projects:
            print("No projects found in your Custom Vision account.")
            return

        print(f"\nFound {len(projects)} project(s):")
        for i, project in enumerate(projects, 1):
            print(f"  {i}. {project.name} (ID: {project.id})")

        # Export all projects
        print("\nStarting export...")
        exporter.export_all(download_images=True)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
