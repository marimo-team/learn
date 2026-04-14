#!/usr/bin/env python
"""Generate a static site from Jinja2 templates and lesson data."""

import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

import frontmatter
import markdown as md
from jinja2 import Environment, FileSystemLoader

from utils import get_notebook_title


def transform_lessons(data: dict, root: Path, branch: str) -> dict:
    """Transform raw lesson data into template-ready form."""
    for course_id, course in data.items():
        desc = course.get("description", "").strip()
        course["description_html"] = f"<p>{desc}</p>" if desc else ""
        course["notebooks"] = [
            {
                "title": get_notebook_title(root / course_id / nb)
                         or re.sub(r"^\d+_", "", nb.replace(".py", "")).replace("_", " ").title(),
                "html_path": f"{course_id}/{nb.replace('.py', '.html')}",
                "local_html_path": nb.replace(".py", ".html"),
                "molab_url": f"https://molab.marimo.io/github/marimo-team/learn/blob/{branch}/{course_id}/{nb}",
            }
            for nb in course.get("notebooks", [])
        ]
        index_md = root / course_id / "index.md"
        post = frontmatter.load(index_md)
        course["body_html"] = md.markdown(post.content, extensions=["fenced_code", "tables"])
    return data


def render(template, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.render(**kwargs))


def main():
    parser = argparse.ArgumentParser(description="Generate static site from lesson data")
    parser.add_argument("--root", required=True, help="Project root directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--data", required=True, help="Path to lessons JSON file")
    parser.add_argument("--branch", required=True, help="Git branch name for molab URLs")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    lessons = transform_lessons(json.loads(Path(args.data).read_text()), root, args.branch)
    env = Environment(loader=FileSystemLoader(root / "templates"))
    current_year = datetime.date.today().year

    render(
        env.get_template("index.html"),
        output / "index.html",
        courses=lessons,
        current_year=current_year,
        root_path="",
    )

    assets_src = root / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, output / "assets", dirs_exist_ok=True)

    for course_id, lesson in lessons.items():
        render(
            env.get_template("lesson.html"),
            output / course_id / "index.html",
            lesson=lesson,
            current_year=current_year,
            root_path="../",
        )

    page_template = env.get_template("page.html")
    for page_src in sorted((root / "pages").glob("*.md")):
        post = frontmatter.load(page_src)
        render(
            page_template,
            output / page_src.stem / "index.html",
            title=post.get("title", page_src.stem),
            body_html=md.markdown(post.content, extensions=["fenced_code", "tables"]),
            current_year=current_year,
            root_path="../",
        )


if __name__ == "__main__":
    main()
