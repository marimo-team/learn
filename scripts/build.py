#!/usr/bin/env python3

import os
import subprocess
import argparse
import json
from typing import List, Dict, Any
from pathlib import Path
import re


def export_html_wasm(notebook_path: str, output_dir: str, as_app: bool = False) -> bool:
    """Export a single marimo notebook to HTML format.

    Returns:
        bool: True if export succeeded, False otherwise
    """
    output_path = notebook_path.replace(".py", ".html")

    cmd = ["marimo", "export", "html-wasm"]
    if as_app:
        print(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])
    else:
        print(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])

    try:
        output_file = os.path.join(output_dir, output_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        cmd.extend([notebook_path, "-o", output_file])
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {notebook_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def get_course_metadata(course_dir: Path) -> Dict[str, Any]:
    """Extract metadata from a course directory."""
    metadata = {
        "id": course_dir.name,
        "title": course_dir.name.replace("_", " ").title(),
        "description": "",
        "notebooks": []
    }
    
    # Try to read README.md for description
    readme_path = course_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract first paragraph as description
            if content:
                lines = content.split("\n")
                # Skip title line if it exists
                start_idx = 1 if lines and lines[0].startswith("#") else 0
                description_lines = []
                for line in lines[start_idx:]:
                    if line.strip() and not line.startswith("#"):
                        description_lines.append(line)
                    elif description_lines:  # Stop at the next heading
                        break
                description = " ".join(description_lines).strip()
                # Clean up the description
                description = description.replace("_", "")
                description = description.replace("[work in progress]", "")
                description = description.replace("(https://github.com/marimo-team/learn/issues/51)", "")
                # Remove any other GitHub issue links
                description = re.sub(r'\[.*?\]\(https://github\.com/.*?/issues/\d+\)', '', description)
                description = re.sub(r'https://github\.com/.*?/issues/\d+', '', description)
                # Clean up any double spaces
                description = re.sub(r'\s+', ' ', description).strip()
                metadata["description"] = description
    
    return metadata


def organize_notebooks_by_course(all_notebooks: List[str]) -> Dict[str, Dict[str, Any]]:
    """Organize notebooks by course."""
    courses = {}
    
    for notebook_path in all_notebooks:
        path = Path(notebook_path)
        course_id = path.parts[0]
        
        if course_id not in courses:
            course_dir = Path(course_id)
            courses[course_id] = get_course_metadata(course_dir)
        
        # Extract notebook info
        filename = path.name
        notebook_id = path.stem
        
        # Try to extract order from filename (e.g., 001_numbers.py -> 1)
        order = 999
        if "_" in notebook_id:
            try:
                order_str = notebook_id.split("_")[0]
                order = int(order_str)
            except ValueError:
                pass
        
        # Create display name by removing order prefix and underscores
        display_name = notebook_id
        if "_" in notebook_id:
            display_name = "_".join(notebook_id.split("_")[1:])
        
        # Convert display name to title case, but handle italics properly
        parts = display_name.split("_")
        formatted_parts = []
        
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == "" and parts[i+1] == "":
                # Skip empty parts that might come from consecutive underscores
                i += 2
                continue
                
            if i + 1 < len(parts) and (parts[i] == "" or parts[i+1] == ""):
                # This is an italics marker
                if parts[i] == "":
                    # Opening italics
                    text_part = parts[i+1].replace("_", " ").title()
                    formatted_parts.append(f"<em>{text_part}</em>")
                    i += 2
                else:
                    # Text followed by italics marker
                    text_part = parts[i].replace("_", " ").title()
                    formatted_parts.append(text_part)
                    i += 1
            else:
                # Regular text
                text_part = parts[i].replace("_", " ").title()
                formatted_parts.append(text_part)
                i += 1
        
        display_name = " ".join(formatted_parts)
        
        courses[course_id]["notebooks"].append({
            "id": notebook_id,
            "path": notebook_path,
            "display_name": display_name,
            "order": order,
            "original_number": notebook_id.split("_")[0] if "_" in notebook_id else ""
        })
    
    # Sort notebooks by order
    for course_id in courses:
        courses[course_id]["notebooks"].sort(key=lambda x: x["order"])
    
    return courses


def generate_eva_css() -> str:
    """Generate Neon Genesis Evangelion inspired CSS with light/dark mode support."""
    return """
    :root {
        /* Dark mode colors (default) */
        --eva-purple: #9a1eb3;
        --eva-green: #1c7361;
        --eva-orange: #ff6600;
        --eva-blue: #0066ff;
        --eva-red: #ff0000;
        --eva-black: #111111;
        --eva-dark: #222222;
        --eva-terminal-bg: rgba(0, 0, 0, 0.85);
        --eva-text: #e0e0e0;
        --eva-border-radius: 4px;
        --eva-transition: all 0.3s ease;
    }
    
    /* Light mode colors */
    [data-theme="light"] {
        --eva-purple: #7209b7;
        --eva-green: #1c7361;
        --eva-orange: #e65100;
        --eva-blue: #0039cb;
        --eva-red: #c62828;
        --eva-black: #f5f5f5;
        --eva-dark: #e0e0e0;
        --eva-terminal-bg: rgba(255, 255, 255, 0.9);
        --eva-text: #333333;
    }
    
    body {
        background-color: var(--eva-black);
        color: var(--eva-text);
        font-family: 'Courier New', monospace;
        margin: 0;
        padding: 0;
        line-height: 1.6;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    
    .eva-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .eva-header {
        border-bottom: 2px solid var(--eva-green);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        background-color: var(--eva-black);
        z-index: 100;
        backdrop-filter: blur(5px);
        padding-top: 1rem;
        transition: background-color 0.3s ease;
    }
    
    [data-theme="light"] .eva-header {
        background-color: rgba(245, 245, 245, 0.95);
    }
    
    .eva-logo {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--eva-green);
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(28, 115, 97, 0.5);
    }
    
    [data-theme="light"] .eva-logo {
        text-shadow: 0 0 10px rgba(28, 115, 97, 0.3);
    }
    
    .eva-nav {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    
    .eva-nav a {
        color: var(--eva-text);
        text-decoration: none;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 1px;
        transition: color 0.3s;
        position: relative;
        padding: 0.5rem 0;
    }
    
    .eva-nav a:hover {
        color: var(--eva-green);
    }
    
    .eva-nav a:hover::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: var(--eva-green);
        animation: scanline 1.5s linear infinite;
    }
    
    .theme-toggle {
        background: none;
        border: none;
        color: var(--eva-text);
        cursor: pointer;
        font-size: 1.2rem;
        padding: 0.5rem;
        margin-left: 1rem;
        transition: color 0.3s;
    }
    
    .theme-toggle:hover {
        color: var(--eva-green);
    }
    
    .eva-hero {
        background-color: var(--eva-terminal-bg);
        border: 1px solid var(--eva-green);
        padding: 3rem 2rem;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border-radius: var(--eva-border-radius);
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        background-image: linear-gradient(45deg, rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.7)), url('https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg');
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    
    [data-theme="light"] .eva-hero {
        background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7)), url('https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg');
    }
    
    .eva-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: var(--eva-green);
        animation: scanline 3s linear infinite;
    }
    
    .eva-hero h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: var(--eva-green);
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(28, 115, 97, 0.5);
    }
    
    [data-theme="light"] .eva-hero h1 {
        text-shadow: 0 0 10px rgba(28, 115, 97, 0.3);
    }
    
    .eva-hero p {
        font-size: 1.1rem;
        max-width: 800px;
        margin-bottom: 2rem;
        line-height: 1.8;
    }
    
    .eva-features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
    }
    
    .eva-feature {
        background-color: var(--eva-terminal-bg);
        border: 1px solid var(--eva-blue);
        padding: 1.5rem;
        border-radius: var(--eva-border-radius);
        transition: var(--eva-transition);
        position: relative;
        overflow: hidden;
    }
    
    .eva-feature:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 102, 255, 0.2);
    }
    
    .eva-feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--eva-blue);
    }
    
    .eva-feature h3 {
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: var(--eva-blue);
    }
    
    .eva-section-title {
        font-size: 2rem;
        color: var(--eva-green);
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        position: relative;
        padding-bottom: 1rem;
    }
    
    .eva-section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 2px;
        background-color: var(--eva-green);
    }
    
    /* Flashcard view for courses */
    .eva-courses {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 2rem;
    }
    
    .eva-course {
        background-color: var(--eva-terminal-bg);
        border: 1px solid var(--eva-purple);
        border-radius: var(--eva-border-radius);
        transition: var(--eva-transition), height 0.4s cubic-bezier(0.19, 1, 0.22, 1);
        position: relative;
        overflow: hidden;
        height: 350px;
        display: flex;
        flex-direction: column;
    }
    
    .eva-course:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(154, 30, 179, 0.3);
    }
    
    .eva-course::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: var(--eva-purple);
        animation: scanline 2s linear infinite;
    }
    
    .eva-course-badge {
        position: absolute;
        top: 15px;
        right: -40px;
        background: linear-gradient(135deg, var(--eva-orange) 0%, #ff9500 100%);
        color: var(--eva-black);
        font-size: 0.65rem;
        padding: 0.3rem 2.5rem;
        text-transform: uppercase;
        font-weight: bold;
        z-index: 3;
        letter-spacing: 1px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        transform: rotate(45deg);
        text-shadow: 0 1px 1px rgba(255, 255, 255, 0.2);
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        border-bottom: 1px solid rgba(0, 0, 0, 0.2);
        white-space: nowrap;
        overflow: hidden;
    }
    
    .eva-course-badge i {
        margin-right: 4px;
        font-size: 0.7rem;
    }
    
    [data-theme="light"] .eva-course-badge {
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        text-shadow: 0 1px 1px rgba(255, 255, 255, 0.4);
    }
    
    .eva-course-badge::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: scanline 2s linear infinite;
    }
    
    .eva-course-header {
        padding: 1rem 1.5rem;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(154, 30, 179, 0.3);
        z-index: 2;
        background-color: var(--eva-terminal-bg);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3.5rem;
        box-sizing: border-box;
    }
    
    .eva-course-title {
        font-size: 1.3rem;
        color: var(--eva-purple);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .eva-course-toggle {
        color: var(--eva-purple);
        font-size: 1.5rem;
        transition: transform 0.4s cubic-bezier(0.19, 1, 0.22, 1);
    }
    
    .eva-course.active .eva-course-toggle {
        transform: rotate(180deg);
    }
    
    .eva-course-front {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 1.5rem;
        margin-top: 3.5rem;
        transition: opacity 0.3s ease, transform 0.3s ease;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: calc(100% - 3.5rem);
        background-color: var(--eva-terminal-bg);
        z-index: 1;
        box-sizing: border-box;
    }
    
    .eva-course.active .eva-course-front {
        opacity: 0;
        transform: translateY(-10px);
        pointer-events: none;
    }
    
    .eva-course-description {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        line-height: 1.6;
        flex-grow: 1;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        max-height: 150px;
    }
    
    .eva-course-stats {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: var(--eva-text);
        opacity: 0.7;
    }
    
    .eva-course-content {
        position: absolute;
        top: 3.5rem;
        left: 0;
        width: 100%;
        height: calc(100% - 3.5rem);
        padding: 1.5rem;
        background-color: var(--eva-terminal-bg);
        transition: opacity 0.3s ease, transform 0.3s ease;
        opacity: 0;
        transform: translateY(10px);
        pointer-events: none;
        overflow-y: auto;
        z-index: 1;
        box-sizing: border-box;
    }
    
    .eva-course.active .eva-course-content {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }
    
    .eva-course.active {
        height: auto;
        min-height: 350px;
        max-height: 800px;
        transition: height 0.4s cubic-bezier(0.19, 1, 0.22, 1), transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .eva-notebooks {
        margin-top: 1rem;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 0.75rem;
    }
    
    .eva-notebook {
        margin-bottom: 0.5rem;
        padding: 0.75rem;
        border-left: 2px solid var(--eva-blue);
        transition: all 0.25s ease;
        display: flex;
        align-items: center;
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 0 var(--eva-border-radius) var(--eva-border-radius) 0;
        opacity: 1;
        transform: translateX(0);
    }
    
    [data-theme="light"] .eva-notebook {
        background-color: rgba(0, 0, 0, 0.05);
    }
    
    .eva-notebook:hover {
        background-color: rgba(0, 102, 255, 0.1);
        padding-left: 1rem;
        transform: translateX(3px);
    }
    
    .eva-notebook a {
        color: var(--eva-text);
        text-decoration: none;
        display: block;
        font-size: 0.9rem;
        flex-grow: 1;
    }
    
    .eva-notebook a:hover {
        color: var(--eva-blue);
    }
    
    .eva-notebook-number {
        color: var(--eva-blue);
        font-size: 0.8rem;
        margin-right: 0.75rem;
        opacity: 0.7;
        min-width: 24px;
        font-weight: bold;
    }
    
    .eva-button {
        display: inline-block;
        background-color: transparent;
        color: var(--eva-green);
        border: 1px solid var(--eva-green);
        padding: 0.7rem 1.5rem;
        text-decoration: none;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 1px;
        transition: var(--eva-transition);
        cursor: pointer;
        border-radius: var(--eva-border-radius);
        position: relative;
        overflow: hidden;
    }
    
    .eva-button:hover {
        background-color: var(--eva-green);
        color: var(--eva-black);
    }
    
    .eva-button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .eva-button:hover::after {
        left: 100%;
    }
    
    .eva-course-button {
        margin-top: 1rem;
        margin-bottom: 1rem;
        align-self: center;
    }
    
    .eva-cta {
        background-color: var(--eva-terminal-bg);
        border: 1px solid var(--eva-orange);
        padding: 3rem 2rem;
        margin: 4rem 0;
        text-align: center;
        border-radius: var(--eva-border-radius);
        position: relative;
        overflow: hidden;
    }
    
    .eva-cta h2 {
        font-size: 2rem;
        color: var(--eva-orange);
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }
    
    .eva-cta p {
        max-width: 600px;
        margin: 0 auto 2rem;
        font-size: 1.1rem;
    }
    
    .eva-cta .eva-button {
        color: var(--eva-orange);
        border-color: var(--eva-orange);
    }
    
    .eva-cta .eva-button:hover {
        background-color: var(--eva-orange);
        color: var(--eva-black);
    }
    
    .eva-footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 2px solid var(--eva-green);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2rem;
    }
    
    .eva-footer-logo {
        max-width: 200px;
        margin-bottom: 1rem;
    }
    
    .eva-footer-links {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .eva-footer-links a {
        color: var(--eva-text);
        text-decoration: none;
        transition: var(--eva-transition);
    }
    
    .eva-footer-links a:hover {
        color: var(--eva-green);
    }
    
    .eva-social-links {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .eva-social-links a {
        color: var(--eva-text);
        font-size: 1.5rem;
        transition: var(--eva-transition);
    }
    
    .eva-social-links a:hover {
        color: var(--eva-green);
        transform: translateY(-3px);
    }
    
    .eva-footer-copyright {
        font-size: 0.9rem;
        text-align: center;
    }
    
    .eva-search {
        position: relative;
        margin-bottom: 3rem;
    }
    
    .eva-search input {
        width: 100%;
        padding: 1rem;
        background-color: var(--eva-terminal-bg);
        border: 1px solid var(--eva-green);
        color: var(--eva-text);
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        border-radius: var(--eva-border-radius);
        outline: none;
        transition: var(--eva-transition);
    }
    
    .eva-search input:focus {
        box-shadow: 0 0 10px rgba(28, 115, 97, 0.3);
    }
    
    .eva-search input::placeholder {
        color: rgba(224, 224, 224, 0.5);
    }
    
    .eva-search-icon {
        position: absolute;
        right: 1rem;
        top: 50%;
        transform: translateY(-50%);
        color: var(--eva-green);
        font-size: 1.2rem;
    }
    
    @keyframes scanline {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
    
    @keyframes blink {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0;
        }
    }
    
    .eva-cursor {
        display: inline-block;
        width: 10px;
        height: 1.2em;
        background-color: var(--eva-green);
        margin-left: 2px;
        animation: blink 1s infinite;
        vertical-align: middle;
    }
    
    @media (max-width: 768px) {
        .eva-courses {
            grid-template-columns: 1fr;
        }
        
        .eva-header {
            flex-direction: column;
            align-items: flex-start;
            padding: 1rem;
        }
        
        .eva-nav {
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        
        .eva-hero {
            padding: 2rem 1rem;
        }
        
        .eva-hero h1 {
            font-size: 2rem;
        }
        
        .eva-features {
            grid-template-columns: 1fr;
        }
        
        .eva-footer {
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        
        .eva-notebooks {
            grid-template-columns: 1fr;
        }
    }
    
    .eva-course.closing .eva-course-content {
        opacity: 0;
        transform: translateY(10px);
        transition: opacity 0.2s ease, transform 0.2s ease;
    }
    
    .eva-course.closing .eva-course-front {
        opacity: 1;
        transform: translateY(0);
        transition: opacity 0.3s ease 0.1s, transform 0.3s ease 0.1s;
    }
    """


def generate_index(courses: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Generate the index.html file with Neon Genesis Evangelion aesthetics."""
    print("Generating index.html")

    index_path = os.path.join(output_dir, "index.html")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(
                """<!DOCTYPE html>
<html lang="en" data-theme="dark">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marimo Learn - Interactive Educational Notebooks</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
""" + generate_eva_css() + """
    </style>
  </head>
<body>
    <div class="eva-container">
        <header class="eva-header">
            <div class="eva-logo">MARIMO LEARN</div>
            <nav class="eva-nav">
                <a href="#features">Features</a>
                <a href="#courses">Courses</a>
                <a href="#contribute">Contribute</a>
                <a href="https://docs.marimo.io" target="_blank">Documentation</a>
                <a href="https://github.com/marimo-team/learn" target="_blank">GitHub</a>
                <button id="themeToggle" class="theme-toggle" aria-label="Toggle dark/light mode">
                    <i class="fas fa-moon"></i>
                </button>
            </nav>
        </header>

        <section class="eva-hero">
            <h1>Interactive Learning with Marimo<span class="eva-cursor"></span></h1>
            <p>
                A curated collection of educational notebooks covering computer science, 
                mathematics, data science, and more. Built with marimo - the reactive 
                Python notebook that makes data exploration delightful.
            </p>
            <a href="#courses" class="eva-button">Explore Courses</a>
        </section>

        <section id="features">
            <h2 class="eva-section-title">Why Marimo Learn?</h2>
            <div class="eva-features">
                <div class="eva-feature">
                    <div class="eva-feature-icon"><i class="fas fa-bolt"></i></div>
                    <h3>Reactive Notebooks</h3>
                    <p>Experience the power of reactive programming with marimo notebooks that automatically update when dependencies change.</p>
                </div>
                <div class="eva-feature">
                    <div class="eva-feature-icon"><i class="fas fa-code"></i></div>
                    <h3>Learn by Doing</h3>
                    <p>Interactive examples and exercises help you understand concepts through hands-on practice.</p>
                </div>
                <div class="eva-feature">
                    <div class="eva-feature-icon"><i class="fas fa-graduation-cap"></i></div>
                    <h3>Comprehensive Courses</h3>
                    <p>From Python basics to advanced optimization techniques, our courses cover a wide range of topics.</p>
                </div>
            </div>
        </section>

        <section id="courses">
            <h2 class="eva-section-title">Explore Courses</h2>
            <div class="eva-search">
                <input type="text" id="courseSearch" placeholder="Search courses and notebooks...">
                <span class="eva-search-icon"><i class="fas fa-search"></i></span>
            </div>
            <div class="eva-courses">
"""
            )
            
            # Define the custom order for courses
            course_order = ["python", "probability", "polars", "optimization", "functional_programming"]
            
            # Create a dictionary of courses by ID for easy lookup
            courses_by_id = {course["id"]: course for course in courses.values()}
            
            # Determine which courses are "work in progress" based on description or notebook count
            work_in_progress = set()
            for course_id, course in courses_by_id.items():
                # Consider a course as "work in progress" if it has few notebooks or contains specific phrases
                if (len(course["notebooks"]) < 5 or 
                    "work in progress" in course["description"].lower() or
                    "help us add" in course["description"].lower() or
                    "check back later" in course["description"].lower()):
                    work_in_progress.add(course_id)
            
            # First output courses in the specified order
            for course_id in course_order:
                if course_id in courses_by_id:
                    course = courses_by_id[course_id]
                    
                    # Skip if no notebooks
                    if not course["notebooks"]:
                        continue
                    
                    # Count notebooks
                    notebook_count = len(course["notebooks"])
                    
                    # Determine if this course is a work in progress
                    is_wip = course_id in work_in_progress
                    
                    f.write(
                        f'<div class="eva-course" data-course-id="{course["id"]}">\n'
                    )
                    
                    # Add WIP badge if needed
                    if is_wip:
                        f.write(f'    <div class="eva-course-badge"><i class="fas fa-code-branch"></i> In Progress</div>\n')
                    
                    f.write(
                        f'    <div class="eva-course-header">\n'
                        f'        <h2 class="eva-course-title">{course["title"]}</h2>\n'
                        f'        <span class="eva-course-toggle"><i class="fas fa-chevron-down"></i></span>\n'
                        f'    </div>\n'
                        f'    <div class="eva-course-front">\n'
                        f'        <p class="eva-course-description">{course["description"]}</p>\n'
                        f'        <div class="eva-course-stats">\n'
                        f'            <span><i class="fas fa-book"></i> {notebook_count} notebook{"s" if notebook_count != 1 else ""}</span>\n'
                        f'        </div>\n'
                        f'        <button class="eva-button eva-course-button">View Notebooks</button>\n'
                        f'    </div>\n'
                        f'    <div class="eva-course-content">\n'
                        f'        <div class="eva-notebooks">\n'
                    )
                    
                    for i, notebook in enumerate(course["notebooks"]):
                        # Use original file number instead of sequential numbering
                        notebook_number = notebook.get("original_number", f"{i+1:02d}")
                        f.write(
                            f'            <div class="eva-notebook">\n'
                            f'                <span class="eva-notebook-number">{notebook_number}</span>\n'
                            f'                <a href="{notebook["path"].replace(".py", ".html")}" data-notebook-title="{notebook["display_name"]}">{notebook["display_name"]}</a>\n'
                            f'            </div>\n'
                        )

                    f.write(
                        f'        </div>\n'
                        f'    </div>\n'
                        f'</div>\n'
                    )
                    
                    # Remove from the dictionary so we don't output it again
                    del courses_by_id[course_id]
            
            # Then output any remaining courses alphabetically
            sorted_remaining_courses = sorted(courses_by_id.values(), key=lambda x: x["title"])
            
            for course in sorted_remaining_courses:
                # Skip if no notebooks
                if not course["notebooks"]:
                    continue
                
                # Count notebooks
                notebook_count = len(course["notebooks"])
                
                # Determine if this course is a work in progress
                is_wip = course["id"] in work_in_progress
                
                f.write(
                    f'<div class="eva-course" data-course-id="{course["id"]}">\n'
                )
                
                # Add WIP badge if needed
                if is_wip:
                    f.write(f'    <div class="eva-course-badge"><i class="fas fa-code-branch"></i> In Progress</div>\n')
                
                f.write(
                    f'    <div class="eva-course-header">\n'
                    f'        <h2 class="eva-course-title">{course["title"]}</h2>\n'
                    f'        <span class="eva-course-toggle"><i class="fas fa-chevron-down"></i></span>\n'
                    f'    </div>\n'
                    f'    <div class="eva-course-front">\n'
                    f'        <p class="eva-course-description">{course["description"]}</p>\n'
                    f'        <div class="eva-course-stats">\n'
                    f'            <span><i class="fas fa-book"></i> {notebook_count} notebook{"s" if notebook_count != 1 else ""}</span>\n'
                    f'        </div>\n'
                    f'        <button class="eva-button eva-course-button">View Notebooks</button>\n'
                    f'    </div>\n'
                    f'    <div class="eva-course-content">\n'
                    f'        <div class="eva-notebooks">\n'
                )
                
                for i, notebook in enumerate(course["notebooks"]):
                    # Use original file number instead of sequential numbering
                    notebook_number = notebook.get("original_number", f"{i+1:02d}")
                    f.write(
                        f'            <div class="eva-notebook">\n'
                        f'                <span class="eva-notebook-number">{notebook_number}</span>\n'
                        f'                <a href="{notebook["path"].replace(".py", ".html")}" data-notebook-title="{notebook["display_name"]}">{notebook["display_name"]}</a>\n'
                        f'            </div>\n'
                    )

                f.write(
                    f'        </div>\n'
                    f'    </div>\n'
                    f'</div>\n'
                )
            
            f.write(
                """            </div>
        </section>

        <section id="contribute" class="eva-cta">
            <h2>Contribute to Marimo Learn</h2>
            <p>
                Help us expand our collection of educational notebooks. Whether you're an expert in machine learning, 
                statistics, or any other field, your contributions are welcome!
            </p>
            <a href="https://github.com/marimo-team/learn" target="_blank" class="eva-button">
                <i class="fab fa-github"></i> Contribute on GitHub
            </a>
        </section>

        <footer class="eva-footer">
            <div class="eva-footer-logo">
                <a href="https://marimo.io" target="_blank">
                    <img src="https://marimo.io/logotype-wide.svg" alt="Marimo" width="200">
                </a>
            </div>
            <div class="eva-social-links">
                <a href="https://github.com/marimo-team" target="_blank" aria-label="GitHub"><i class="fab fa-github"></i></a>
                <a href="https://marimo.io/discord?ref=learn" target="_blank" aria-label="Discord"><i class="fab fa-discord"></i></a>
                <a href="https://twitter.com/marimo_io" target="_blank" aria-label="Twitter"><i class="fab fa-twitter"></i></a>
                <a href="https://www.youtube.com/@marimo-io" target="_blank" aria-label="YouTube"><i class="fab fa-youtube"></i></a>
                <a href="https://www.linkedin.com/company/marimo-io" target="_blank" aria-label="LinkedIn"><i class="fab fa-linkedin"></i></a>
            </div>
            <div class="eva-footer-links">
                <a href="https://marimo.io" target="_blank">Website</a>
                <a href="https://docs.marimo.io" target="_blank">Documentation</a>
                <a href="https://github.com/marimo-team/learn" target="_blank">GitHub</a>
            </div>
            <div class="eva-footer-copyright">
                © 2025 Marimo Inc. All rights reserved.
            </div>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Theme toggle functionality
            const themeToggle = document.getElementById('themeToggle');
            const themeIcon = themeToggle.querySelector('i');
            
            // Check for saved theme preference or use default (dark)
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeIcon(savedTheme);
            
            // Toggle theme when button is clicked
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateThemeIcon(newTheme);
            });
            
            function updateThemeIcon(theme) {
                if (theme === 'dark') {
                    themeIcon.className = 'fas fa-sun';
                } else {
                    themeIcon.className = 'fas fa-moon';
                }
            }
            
            // Terminal typing effect for hero text
            const heroTitle = document.querySelector('.eva-hero h1');
            const heroText = document.querySelector('.eva-hero p');
            const cursor = document.querySelector('.eva-cursor');
            
            const originalTitle = heroTitle.textContent;
            const originalText = heroText.textContent.trim();
            
            heroTitle.textContent = '';
            heroText.textContent = '';
            
            let titleIndex = 0;
            let textIndex = 0;
            
            function typeTitle() {
                if (titleIndex < originalTitle.length) {
                    heroTitle.textContent += originalTitle.charAt(titleIndex);
                    titleIndex++;
                    setTimeout(typeTitle, 50);
                } else {
                    cursor.style.display = 'none';
                    setTimeout(typeText, 500);
                }
            }
            
            function typeText() {
                if (textIndex < originalText.length) {
                    heroText.textContent += originalText.charAt(textIndex);
                    textIndex++;
                    setTimeout(typeText, 20);
                }
            }
            
            typeTitle();
            
            // Course toggle functionality - flashcard style
            const courseHeaders = document.querySelectorAll('.eva-course-header');
            const courseButtons = document.querySelectorAll('.eva-course-button');
            
            // Function to toggle course
            function toggleCourse(course) {
                const isActive = course.classList.contains('active');
                
                // First close all courses with a slight delay for better visual effect
                document.querySelectorAll('.eva-course.active').forEach(c => {
                    if (c !== course) {
                        // Add a closing class for animation
                        c.classList.add('closing');
                        // Remove active class after a short delay
                        setTimeout(() => {
                            c.classList.remove('active');
                            c.classList.remove('closing');
                        }, 300);
                    }
                });
                
                // Toggle the clicked course
                if (!isActive) {
                    // Add a small delay before opening to allow others to close
                    setTimeout(() => {
                        course.classList.add('active');
                        
                        // Check if the course has any notebooks
                        const notebooks = course.querySelectorAll('.eva-notebook');
                        const content = course.querySelector('.eva-course-content');
                        
                        if (notebooks.length === 0 && !content.querySelector('.eva-empty-message')) {
                            // If no notebooks, show a message
                            const emptyMessage = document.createElement('p');
                            emptyMessage.className = 'eva-empty-message';
                            emptyMessage.textContent = 'No notebooks available in this course yet.';
                            emptyMessage.style.color = 'var(--eva-text)';
                            emptyMessage.style.fontStyle = 'italic';
                            emptyMessage.style.opacity = '0.7';
                            emptyMessage.style.textAlign = 'center';
                            emptyMessage.style.padding = '1rem 0';
                            content.appendChild(emptyMessage);
                        }
                        
                        // Animate notebooks to appear sequentially
                        notebooks.forEach((notebook, index) => {
                            notebook.style.opacity = '0';
                            notebook.style.transform = 'translateX(-10px)';
                            setTimeout(() => {
                                notebook.style.opacity = '1';
                                notebook.style.transform = 'translateX(0)';
                            }, 50 + (index * 30)); // Stagger the animations
                        });
                    }, 100);
                }
            }
            
            // Add click event to course headers
            courseHeaders.forEach(header => {
                header.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    const currentCourse = this.closest('.eva-course');
                    toggleCourse(currentCourse);
                });
            });
            
            // Add click event to course buttons
            courseButtons.forEach(button => {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    const currentCourse = this.closest('.eva-course');
                    toggleCourse(currentCourse);
                });
            });
            
            // Search functionality with improved matching
            const searchInput = document.getElementById('courseSearch');
            const courses = document.querySelectorAll('.eva-course');
            const notebooks = document.querySelectorAll('.eva-notebook');
            
            searchInput.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                
                if (searchTerm === '') {
                    // Reset all visibility
                    courses.forEach(course => {
                        course.style.display = 'block';
                        course.classList.remove('active');
                    });
                    
                    notebooks.forEach(notebook => {
                        notebook.style.display = 'flex';
                    });
                    
                    // Open the first course with notebooks by default when search is cleared
                    for (let i = 0; i < courses.length; i++) {
                        const courseNotebooks = courses[i].querySelectorAll('.eva-notebook');
                        if (courseNotebooks.length > 0) {
                            courses[i].classList.add('active');
                            break;
                        }
                    }
                    
                    return;
                }
                
                // First hide all courses
                courses.forEach(course => {
                    course.style.display = 'none';
                    course.classList.remove('active');
                });
                
                // Then show courses and notebooks that match the search
                let hasResults = false;
                
                // Track which courses have matching notebooks
                const coursesWithMatchingNotebooks = new Set();
                
                // First check notebooks
                notebooks.forEach(notebook => {
                    const notebookTitle = notebook.querySelector('a').getAttribute('data-notebook-title').toLowerCase();
                    const matchesSearch = notebookTitle.includes(searchTerm);
                    
                    notebook.style.display = matchesSearch ? 'flex' : 'none';
                    
                    if (matchesSearch) {
                        const course = notebook.closest('.eva-course');
                        coursesWithMatchingNotebooks.add(course.getAttribute('data-course-id'));
                        hasResults = true;
                    }
                });
                
                // Then check course titles and descriptions
                courses.forEach(course => {
                    const courseId = course.getAttribute('data-course-id');
                    const courseTitle = course.querySelector('.eva-course-title').textContent.toLowerCase();
                    const courseDescription = course.querySelector('.eva-course-description').textContent.toLowerCase();
                    
                    const courseMatches = courseTitle.includes(searchTerm) || courseDescription.includes(searchTerm);
                    
                    // Show course if it matches or has matching notebooks
                    if (courseMatches || coursesWithMatchingNotebooks.has(courseId)) {
                        course.style.display = 'block';
                        course.classList.add('active');
                        hasResults = true;
                        
                        // If course matches but doesn't have matching notebooks, show all its notebooks
                        if (courseMatches && !coursesWithMatchingNotebooks.has(courseId)) {
                            course.querySelectorAll('.eva-notebook').forEach(nb => {
                                nb.style.display = 'flex';
                            });
                        }
                    }
                });
            });
            
            // Open the first course with notebooks by default
            let firstCourseWithNotebooks = null;
            for (let i = 0; i < courses.length; i++) {
                const courseNotebooks = courses[i].querySelectorAll('.eva-notebook');
                if (courseNotebooks.length > 0) {
                    firstCourseWithNotebooks = courses[i];
                    break;
                }
            }
            
            if (firstCourseWithNotebooks) {
                firstCourseWithNotebooks.classList.add('active');
            } else if (courses.length > 0) {
                // If no courses have notebooks, just open the first one
                courses[0].classList.add('active');
            }
            
            // Smooth scrolling for anchor links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    const targetId = this.getAttribute('href');
                    const targetElement = document.querySelector(targetId);
                    
                    if (targetElement) {
                        window.scrollTo({
                            top: targetElement.offsetTop - 100,
                            behavior: 'smooth'
                        });
                    }
                });
            });
        });
    </script>
  </body>
</html>"""
            )
    except IOError as e:
        print(f"Error generating index.html: {e}")


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
    
    # Generate index with organized courses
    generate_index(courses, args.output_dir)
    
    # Save course data as JSON for potential use by other tools
    courses_json_path = os.path.join(args.output_dir, "courses.json")
    with open(courses_json_path, "w", encoding="utf-8") as f:
        json.dump(courses, f, indent=2)
    
    print(f"Build complete! Site generated in {args.output_dir}")
    print(f"Successfully exported {len(successful_notebooks)} out of {len(all_notebooks)} notebooks")


if __name__ == "__main__":
    main()
