#!/usr/bin/env python3

import os
import subprocess
import argparse
import json
import datetime
from datetime import date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from jinja2 import Environment, FileSystemLoader


def export_html_wasm(notebook_path: str, output_dir: str, as_app: bool = False) -> bool:
    """Export a single marimo notebook to HTML format.
    
    Args:
        notebook_path: Path to the notebook to export
        output_dir: Directory to write the output HTML files
        as_app: If True, export as app instead of notebook
    
    Returns:
        bool: True if export succeeded, False otherwise
    """
    # Create directory for the output
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the output path (preserving directory structure)
    rel_path = os.path.basename(os.path.dirname(notebook_path))
    if rel_path != os.path.dirname(notebook_path):
        # Create subdirectory if needed
        os.makedirs(os.path.join(output_dir, rel_path), exist_ok=True)
    
    # Determine output filename (same as input but with .html extension)
    output_filename = os.path.basename(notebook_path).replace(".py", ".html")
    output_path = os.path.join(output_dir, rel_path, output_filename)
    
    # Run marimo export command
    mode = "--mode app" if as_app else "--mode edit"
    cmd = f"marimo export html-wasm {mode} {notebook_path} -o {output_path} --sandbox"
    print(f"Exporting {notebook_path} to {rel_path}/{output_filename} as {'app' if as_app else 'notebook'}")
    print(f"Running command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"Successfully exported {notebook_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {notebook_path}: {e}")
        print(f"Command output: {e.output}")
        return False


def get_course_metadata(course_dir: Path) -> Dict[str, Any]:
    """Extract metadata from a course directory.
    
    Reads the README.md file to extract title and description.
    
    Args:
        course_dir: Path to the course directory
    
    Returns:
        Dict: Dictionary containing course metadata (title, description)
    """
    readme_path = course_dir / "README.md"
    title = course_dir.name.replace("_", " ").title()
    description = ""
    
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Try to extract title from first heading
            title_match = content.split("\n")[0]
            if title_match.startswith("# "):
                title = title_match[2:].strip()
            
            # Extract description from content after first heading
            desc_content = "\n".join(content.split("\n")[1:]).strip()
            if desc_content:
                # Take first paragraph as description
                description = desc_content.split("\n\n")[0].replace("\n", " ").strip()
    
    return {
        "title": title,
        "description": description
    }


def organize_notebooks_by_course(all_notebooks: List[str]) -> Dict[str, Dict[str, Any]]:
    """Organize notebooks by course.
    
    Args:
        all_notebooks: List of paths to notebooks
        
    Returns:
        Dict: A dictionary where keys are course directories and values are
              metadata about the course and its notebooks
    """
    courses = {}
    
    for notebook_path in sorted(all_notebooks):
        # Parse the path to determine course
        # The first directory in the path is the course
        path_parts = Path(notebook_path).parts
        
        if len(path_parts) < 2:
            print(f"Skipping notebook with invalid path: {notebook_path}")
            continue
        
        course_id = path_parts[0]
        
        # If this is a new course, initialize it
        if course_id not in courses:
            course_metadata = get_course_metadata(Path(course_id))
            
            courses[course_id] = {
                "id": course_id,
                "title": course_metadata["title"],
                "description": course_metadata["description"],
                "notebooks": []
            }
        
        # Extract the notebook number and name from the filename
        filename = Path(notebook_path).name
        basename = filename.replace(".py", "")
        
        # Extract notebook metadata
        notebook_title = basename.replace("_", " ").title()
        
        # Try to extract a sequence number from the start of the filename
        # Match patterns like: 01_xxx, 1_xxx, etc.
        import re
        number_match = re.match(r'^(\d+)(?:[_-]|$)', basename)
        notebook_number = number_match.group(1) if number_match else None
        
        # If we found a number, remove it from the title
        if number_match:
            notebook_title = re.sub(r'^\d+\s*[_-]?\s*', '', notebook_title)
        
        # Calculate the HTML output path (for linking)
        html_path = f"{course_id}/{filename.replace('.py', '.html')}"
        
        # Add the notebook to the course
        courses[course_id]["notebooks"].append({
            "path": notebook_path,
            "html_path": html_path,
            "title": notebook_title,
            "display_name": notebook_title,
            "original_number": notebook_number
        })
    
    # Sort notebooks by number if available, otherwise by title
    for course_id, course_data in courses.items():
        # Sort the notebooks list by number and title
        course_data["notebooks"] = sorted(
            course_data["notebooks"],
            key=lambda x: (
                int(x["original_number"]) if x["original_number"] is not None else float('inf'),
                x["title"]
            )
        )
    
    return courses


def generate_clean_tailwind_landing_page(courses: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Generate a clean tailwindcss landing page with green accents.
    
    This generates a modern, minimal landing page for marimo notebooks using tailwindcss.
    The page is designed with clean aesthetics and green color accents using Jinja2 templates.
    
    Args:
        courses: Dictionary of courses metadata
        output_dir: Directory to write the output index.html file
    """
    print("Generating clean tailwindcss landing page")
    
    index_path = os.path.join(output_dir, "index.html")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Jinja2 template
    current_dir = Path(__file__).parent
    templates_dir = current_dir / "templates"
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('index.html')
    
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            # Render the template with the provided data
            rendered_html = template.render(
                courses=courses, 
                current_year=datetime.date.today().year
            )
            f.write(rendered_html)
            
        print(f"Successfully generated clean tailwindcss landing page at {index_path}")
            
    except IOError as e:
        print(f"Error generating clean tailwindcss landing page: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build marimo notebooks")
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    parser.add_argument(
        "--course-dirs", nargs="+", default=None, 
        help="Specific course directories to build (default: all directories with .py files)"
    )
    args = parser.parse_args()

    # Find all course directories (directories containing .py files)
    all_notebooks: List[str] = []
    
    # Directories to exclude from course detection
    excluded_dirs = ["scripts", "env", "__pycache__", ".git", ".github", "assets"]
    
    if args.course_dirs:
        course_dirs = args.course_dirs
    else:
        # Automatically detect course directories (any directory with .py files)
        course_dirs = []
        for item in os.listdir("."):
            if (os.path.isdir(item) and 
                not item.startswith(".") and 
                not item.startswith("_") and
                item not in excluded_dirs):
                # Check if directory contains .py files
                if list(Path(item).glob("*.py")):
                    course_dirs.append(item)
    
    print(f"Found course directories: {', '.join(course_dirs)}")
    
    for directory in course_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        notebooks = [str(path) for path in dir_path.rglob("*.py") 
                    if not path.name.startswith("_") and "/__pycache__/" not in str(path)]
        all_notebooks.extend(notebooks)

    if not all_notebooks:
        print("No notebooks found!")
        return

    # Export notebooks sequentially
    successful_notebooks = []
    for nb in all_notebooks:
        # Determine if notebook should be exported as app or notebook
        # For now, export all as notebooks
        if export_html_wasm(nb, args.output_dir, as_app=False):
            successful_notebooks.append(nb)

    # Organize notebooks by course (only include successfully exported notebooks)
    courses = organize_notebooks_by_course(successful_notebooks)
    
    # Generate landing page using Tailwind CSS
    generate_clean_tailwind_landing_page(courses, args.output_dir)
    
    # Save course data as JSON for potential use by other tools
    courses_json_path = os.path.join(args.output_dir, "courses.json")
    with open(courses_json_path, "w", encoding="utf-8") as f:
        json.dump(courses, f, indent=2)
    
    print(f"Build complete! Site generated in {args.output_dir}")
    print(f"Successfully exported {len(successful_notebooks)} out of {len(all_notebooks)} notebooks")


if __name__ == "__main__":
    main()
