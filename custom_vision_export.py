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

    def load_existing_metadata(self, project_dir: Path) -> dict:
        """Load existing metadata if available."""
        metadata_file = project_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def is_image_complete(self, existing_img: dict, new_img, images_dir: Path) -> bool:
        """Check if an image's data is already complete (file + metadata)."""
        if not existing_img:
            return False

        # Check if image file exists
        local_filename = existing_img.get("local_filename")
        if not local_filename:
            return False

        image_path = images_dir / local_filename
        if not image_path.exists():
            return False

        # Check if tags match (compare tag IDs)
        existing_tag_ids = set(t["tag_id"] for t in existing_img.get("tags", []))
        new_tag_ids = set(str(t.tag_id) for t in (new_img.tags or []))
        if existing_tag_ids != new_tag_ids:
            return False

        # Check if regions match (compare region count and tag IDs)
        existing_region_tags = sorted(r["tag_id"] for r in existing_img.get("regions", []))
        new_region_tags = sorted(str(r.tag_id) for r in (new_img.regions or []))
        if existing_region_tags != new_region_tags:
            return False

        return True

    def export_project(self, project, download_images: bool = True) -> dict:
        """Export all data from a single project."""
        project_id = str(project.id)
        project_name = project.name

        # Create project directory
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in project_name)
        project_dir = self.output_dir / safe_name
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting project: {project_name}")

        # Load existing metadata if available
        existing_data = self.load_existing_metadata(project_dir)
        existing_images_by_id = {}
        if existing_data:
            existing_images_by_id = {img["id"]: img for img in existing_data.get("images", [])}
            print(f"  Found existing metadata with {len(existing_images_by_id)} images")

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

        skipped_count = 0
        downloaded_count = 0
        metadata_updated_count = 0

        for image in tqdm(all_images, desc="  Processing images"):
            image_id = str(image.id)
            existing_img = existing_images_by_id.get(image_id)

            # Check if this image is already complete
            if existing_img and self.is_image_complete(existing_img, image, images_dir):
                # Use existing metadata as-is
                images_metadata.append(existing_img)
                skipped_count += 1
                continue

            # Need to process this image (new or updated)
            metadata = self.export_image_metadata(image, tags_lookup)

            if existing_img and existing_img.get("local_filename"):
                metadata_updated_count += 1

            if download_images and image.original_image_uri:
                # Determine file extension from URL or default to jpg
                ext = ".jpg"
                if image.original_image_uri:
                    url_path = image.original_image_uri.split('?')[0]
                    if '.' in url_path.split('/')[-1]:
                        ext = '.' + url_path.split('.')[-1].lower()

                image_filename = f"{image.id}{ext}"
                image_path = images_dir / image_filename

                # Skip download if image file already exists
                if image_path.exists():
                    metadata["local_filename"] = image_filename
                elif self.download_image(image.original_image_uri, image_path):
                    metadata["local_filename"] = image_filename
                    downloaded_count += 1
                else:
                    metadata["local_filename"] = None
                    metadata["download_error"] = True

            images_metadata.append(metadata)

        if skipped_count > 0:
            print(f"  Skipped {skipped_count} already complete images")
        if metadata_updated_count > 0:
            print(f"  Updated metadata for {metadata_updated_count} images")
        if downloaded_count > 0:
            print(f"  Downloaded {downloaded_count} new images")

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

        # Load existing summary to merge with
        summary_file = self.output_dir / "export_summary.json"
        existing_projects_by_id = {}
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
                    existing_projects_by_id = {p["id"]: p for p in existing_summary.get("projects", [])}
                    print(f"  Merging with existing summary ({len(existing_projects_by_id)} projects)")
            except (json.JSONDecodeError, IOError):
                pass

        # Update with new projects (add or replace)
        for p in projects:
            existing_projects_by_id[str(p.id)] = {
                "name": p.name,
                "id": str(p.id),
                "image_count": all_exports[p.name]["summary"]["total_images"]
            }

        # Save merged summary
        merged_projects = list(existing_projects_by_id.values())
        summary = {
            "total_projects": len(merged_projects),
            "projects": merged_projects
        }

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
