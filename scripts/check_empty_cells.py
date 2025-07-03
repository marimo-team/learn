#!/usr/bin/env python3
"""
Script to detect empty cells in marimo notebooks.

An empty cell is defined as:
@app.cell
def _():
    return

This script will:
1. Find all Python files in the repository
2. Check if they contain marimo app definitions
3. Look for empty cell patterns
4. Report any empty cells found
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def is_marimo_notebook(file_path: Path) -> bool:
    """Check if a Python file is a marimo notebook."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for marimo app defn
            return 'marimo.App' in content and '@app.cell' in content
    except (UnicodeDecodeError, IOError):
        return False


def find_empty_cells(file_path: Path) -> List[Tuple[int, str]]:
    """Find empty cells in a marimo notebook."""
    empty_cells = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (UnicodeDecodeError, IOError):
        return empty_cells
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # if line starts with @app.cell decorator
        if line.startswith('@app.cell'):
            # look for the function definition on the next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            if j < len(lines) and lines[j].strip() == 'def _():':
                # found function definition, now look for return statement
                k = j + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                
                if k < len(lines) and lines[k].strip() == 'return':
                    # if any content after return (before next @app.cell or end of file)
                    has_content = False
                    m = k + 1
                    while m < len(lines):
                        line_content = lines[m].strip()
                        if line_content.startswith('@app.cell'):
                            break
                        if line_content and not line_content.startswith('#'):
                            has_content = True
                            break
                        m += 1
                    
                    if not has_content:
                        empty_cells.append((i + 1, lines[i].strip()))
                
                i = k + 1
            else:
                i += 1
        else:
            i += 1
    
    return empty_cells


def main():
    """Main function to check for empty cells."""
    print("🔍 Checking for empty cells in marimo notebooks...")
    
    python_files = []
    for root, dirs, files in os.walk('.'):
        # skip hidden directories and common build/cache
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'build', 'dist']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    marimo_notebooks = []
    for file_path in python_files:
        if is_marimo_notebook(file_path):
            marimo_notebooks.append(file_path)
    
    print(f"📝 Found {len(marimo_notebooks)} marimo notebooks")
    
    # each notebook checked for empty cells
    total_empty_cells = 0
    files_with_empty_cells = []
    
    for notebook_path in marimo_notebooks:
        empty_cells = find_empty_cells(notebook_path)
        if empty_cells:
            total_empty_cells += len(empty_cells)
            files_with_empty_cells.append((notebook_path, empty_cells))
            print(f"❌ {notebook_path}: {len(empty_cells)} empty cell(s)")
            for line_num, line_content in empty_cells:
                print(f"   Line {line_num}: {line_content}")
    
    if files_with_empty_cells:
        print(f"\n💥 Found {total_empty_cells} empty cells in {len(files_with_empty_cells)} files")
        print("\nEmpty cells should be removed or contain meaningful content.")
        print("An empty cell looks like:")
        print("@app.cell")
        print("def _():")
        print("    return")
        print("\nConsider either:")
        print("1. Removing the empty cell entirely")
        print("2. Adding meaningful content to the cell")
        print("3. Adding a comment explaining why the cell is empty")
        
        sys.exit(1)
    else:
        print("✅ No empty cells found!")
        sys.exit(0)


if __name__ == "__main__":
    main() 