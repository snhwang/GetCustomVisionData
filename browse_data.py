"""
Custom Vision Data Browser

A simple interface to browse and explore exported Custom Vision datasets.
"""

import json
import os
from pathlib import Path
from typing import Optional


class DataBrowser:
    """Browse exported Custom Vision data."""

    def __init__(self, export_dir: str = "custom_vision_export"):
        self.export_dir = Path(export_dir)

    def list_projects(self) -> list:
        """List all exported projects."""
        if not self.export_dir.exists():
            return []

        projects = []
        for item in self.export_dir.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    projects.append({
                        "name": data["project"]["name"],
                        "path": str(item),
                        "image_count": data["summary"]["total_images"],
                        "tag_count": data["summary"]["total_tags"],
                        "images_with_regions": data["summary"]["images_with_regions"]
                    })
        return projects

    def load_project(self, project_path: str) -> dict:
        """Load a project's metadata."""
        metadata_file = Path(project_path) / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_tags(self, project_path: str) -> list:
        """Get all tags in a project."""
        data = self.load_project(project_path)
        return data["tags"]

    def get_images_by_tag(self, project_path: str, tag_name: str) -> list:
        """Get all images with a specific tag."""
        data = self.load_project(project_path)
        matching = []
        for img in data["images"]:
            tag_names = [t["tag_name"] for t in img["tags"]]
            region_tags = [r["tag_name"] for r in img["regions"]]
            if tag_name in tag_names or tag_name in region_tags:
                matching.append(img)
        return matching

    def get_images_with_regions(self, project_path: str) -> list:
        """Get all images that have region annotations (object detection)."""
        data = self.load_project(project_path)
        return [img for img in data["images"] if img["regions"]]

    def search_images(self, project_path: str, query: str) -> list:
        """Search images by tag name."""
        data = self.load_project(project_path)
        query_lower = query.lower()
        matching = []
        for img in data["images"]:
            tag_names = [t["tag_name"].lower() for t in img["tags"]]
            region_tags = [r["tag_name"].lower() for r in img["regions"]]
            all_tags = tag_names + region_tags
            if any(query_lower in tag for tag in all_tags):
                matching.append(img)
        return matching

    def get_image_path(self, project_path: str, image_metadata: dict) -> Optional[Path]:
        """Get the local file path for an image."""
        if image_metadata.get("local_filename"):
            return Path(project_path) / "images" / image_metadata["local_filename"]
        return None

    def print_summary(self):
        """Print a summary of all exported data."""
        projects = self.list_projects()

        if not projects:
            print("No exported data found.")
            print(f"Run custom_vision_export.py first to export your data.")
            return

        print("=" * 60)
        print("Exported Custom Vision Data")
        print("=" * 60)

        for project in projects:
            print(f"\nProject: {project['name']}")
            print(f"  Path: {project['path']}")
            print(f"  Images: {project['image_count']}")
            print(f"  Tags: {project['tag_count']}")
            print(f"  Images with ROIs: {project['images_with_regions']}")

            # Load and show tags
            data = self.load_project(project['path'])
            if data["tags"]:
                print("  Tags:")
                for tag in data["tags"]:
                    print(f"    - {tag['name']} ({tag['image_count']} images)")

    def export_coco_format(self, project_path: str, output_file: str = None) -> dict:
        """Export annotations in COCO format for object detection."""
        data = self.load_project(project_path)

        # Create category mapping
        categories = []
        cat_name_to_id = {}
        for i, tag in enumerate(data["tags"], 1):
            categories.append({
                "id": i,
                "name": tag["name"],
                "supercategory": "object"
            })
            cat_name_to_id[tag["name"]] = i

        # Create images and annotations
        images = []
        annotations = []
        ann_id = 1

        for img_idx, img in enumerate(data["images"], 1):
            if not img.get("local_filename"):
                continue

            images.append({
                "id": img_idx,
                "file_name": img["local_filename"],
                "width": img["width"],
                "height": img["height"]
            })

            # Add region annotations
            for region in img["regions"]:
                # Convert normalized coords to pixel coords
                x = region["left"] * img["width"]
                y = region["top"] * img["height"]
                w = region["width"] * img["width"]
                h = region["height"] * img["height"]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_name_to_id.get(region["tag_name"], 0),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        if output_file:
            output_path = Path(project_path) / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2)
            print(f"COCO format saved to: {output_path}")

        return coco_data

    def export_yolo_format(self, project_path: str, output_dir: str = "yolo_labels") -> None:
        """Export annotations in YOLO format for object detection."""
        data = self.load_project(project_path)

        # Create class mapping
        class_names = [tag["name"] for tag in data["tags"]]
        class_to_id = {name: i for i, name in enumerate(class_names)}

        # Create output directory
        labels_dir = Path(project_path) / output_dir
        labels_dir.mkdir(exist_ok=True)

        # Write class names file
        with open(labels_dir / "classes.txt", 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Write label files
        for img in data["images"]:
            if not img.get("local_filename") or not img["regions"]:
                continue

            # Create label filename (same as image but .txt)
            base_name = Path(img["local_filename"]).stem
            label_file = labels_dir / f"{base_name}.txt"

            with open(label_file, 'w') as f:
                for region in img["regions"]:
                    class_id = class_to_id.get(region["tag_name"], 0)

                    # YOLO format: class_id center_x center_y width height (all normalized)
                    center_x = region["left"] + region["width"] / 2
                    center_y = region["top"] + region["height"] / 2

                    f.write(f"{class_id} {center_x} {center_y} {region['width']} {region['height']}\n")

        print(f"YOLO format saved to: {labels_dir}")
        print(f"Class names: {labels_dir / 'classes.txt'}")


def interactive_browser():
    """Run an interactive browsing session."""
    browser = DataBrowser()

    while True:
        print("\n" + "=" * 60)
        print("Custom Vision Data Browser")
        print("=" * 60)
        print("1. Show summary of all projects")
        print("2. List tags in a project")
        print("3. Search images by tag")
        print("4. Export to COCO format")
        print("5. Export to YOLO format")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            browser.print_summary()

        elif choice == "2":
            projects = browser.list_projects()
            if not projects:
                print("No projects found. Run export first.")
                continue

            print("\nProjects:")
            for i, p in enumerate(projects, 1):
                print(f"  {i}. {p['name']}")

            try:
                idx = int(input("Select project number: ")) - 1
                if 0 <= idx < len(projects):
                    tags = browser.get_tags(projects[idx]["path"])
                    print(f"\nTags in {projects[idx]['name']}:")
                    for tag in tags:
                        print(f"  - {tag['name']}: {tag['image_count']} images")
            except (ValueError, IndexError):
                print("Invalid selection")

        elif choice == "3":
            projects = browser.list_projects()
            if not projects:
                print("No projects found.")
                continue

            print("\nProjects:")
            for i, p in enumerate(projects, 1):
                print(f"  {i}. {p['name']}")

            try:
                idx = int(input("Select project number: ")) - 1
                if 0 <= idx < len(projects):
                    query = input("Enter tag to search: ").strip()
                    results = browser.search_images(projects[idx]["path"], query)
                    print(f"\nFound {len(results)} images matching '{query}':")
                    for img in results[:10]:  # Show first 10
                        tags = [t["tag_name"] for t in img["tags"]]
                        print(f"  - {img.get('local_filename', img['id'])}: {', '.join(tags)}")
                    if len(results) > 10:
                        print(f"  ... and {len(results) - 10} more")
            except (ValueError, IndexError):
                print("Invalid selection")

        elif choice == "4":
            projects = browser.list_projects()
            if not projects:
                print("No projects found.")
                continue

            print("\nProjects:")
            for i, p in enumerate(projects, 1):
                print(f"  {i}. {p['name']}")

            try:
                idx = int(input("Select project number: ")) - 1
                if 0 <= idx < len(projects):
                    browser.export_coco_format(projects[idx]["path"], "coco_annotations.json")
            except (ValueError, IndexError):
                print("Invalid selection")

        elif choice == "5":
            projects = browser.list_projects()
            if not projects:
                print("No projects found.")
                continue

            print("\nProjects:")
            for i, p in enumerate(projects, 1):
                print(f"  {i}. {p['name']}")

            try:
                idx = int(input("Select project number: ")) - 1
                if 0 <= idx < len(projects):
                    browser.export_yolo_format(projects[idx]["path"])
            except (ValueError, IndexError):
                print("Invalid selection")

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        browser = DataBrowser()
        browser.print_summary()
    else:
        interactive_browser()
