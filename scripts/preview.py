#!/usr/bin/env python3

import os
import subprocess
import argparse
import webbrowser
import time
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build and preview marimo notebooks site")
    parser.add_argument(
        "--port", default=8000, type=int, help="Port to run the server on"
    )
    parser.add_argument(
        "--no-build", action="store_true", help="Skip building the site (just serve existing files)"
    )
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    args = parser.parse_args()
    
    # Store the current directory
    original_dir = os.getcwd()
    
    try:
        # Build the site if not skipped
        if not args.no_build:
            print("Building site...")
            build_script = Path("scripts/build.py")
            if not build_script.exists():
                print(f"Error: Build script not found at {build_script}")
                return 1
                
            result = subprocess.run(
                [sys.executable, str(build_script), "--output-dir", args.output_dir], 
                check=False
            )
            if result.returncode != 0:
                print("Warning: Build process completed with errors.")
        
        # Check if the output directory exists
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: Output directory '{args.output_dir}' does not exist.")
            return 1
            
        # Change to the output directory
        os.chdir(args.output_dir)
        
        # Open the browser
        url = f"http://localhost:{args.port}"
        print(f"Opening {url} in your browser...")
        webbrowser.open(url)
        
        # Start the server
        print(f"Starting server on port {args.port}...")
        print("Press Ctrl+C to stop the server")
        
        # Use the appropriate Python executable
        subprocess.run([sys.executable, "-m", "http.server", str(args.port)])
        
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Always return to the original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())
