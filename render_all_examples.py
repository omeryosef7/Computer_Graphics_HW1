#!/usr/bin/env python3
"""
Wrapper script to render all example scenes in raytracer/Examples/
Saves rendered images to raytracer/Rendered/
Supports parallel rendering for faster execution.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def render_scene(scene_file, output_image, ray_tracer_script, width=500, height=500):
    """
    Render a single scene file.

    Args:
        scene_file: Path to the scene .txt file
        output_image: Path where the output PNG should be saved
        ray_tracer_script: Path to the ray_tracer.py script
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple of (scene_name, success, message)
    """
    scene_name = scene_file.stem

    cmd = [
        sys.executable,
        str(ray_tracer_script),
        str(scene_file),
        str(output_image),
        "--width", str(width),
        "--height", str(height)
    ]

    try:
        print(f"\n{'='*70}")
        print(f"Starting: {scene_name}")
        print(f"{'='*70}")
        result = subprocess.run(cmd, check=True, capture_output=False)
        return (scene_name, True, f"✓ Successfully rendered {scene_name}.png")
    except subprocess.CalledProcessError as e:
        error_msg = f"✗ Failed to render {scene_name}: {e}"
        return (scene_name, False, error_msg)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Render all example scenes with optional parallel processing'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=f'Number of parallel workers (default: {min(4, os.cpu_count() or 1)})'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=500,
        help='Image width (default: 500)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=500,
        help='Image height (default: 500)'
    )
    args = parser.parse_args()

    # Define directories
    base_dir = Path(__file__).parent
    examples_dir = base_dir / "raytracer" / "Examples"
    output_dir = base_dir / "raytracer" / "Rendered"
    ray_tracer_script = base_dir / "raytracer" / "ray_tracer.py"

    # Verify examples directory exists
    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        sys.exit(1)

    # Verify ray tracer script exists
    if not ray_tracer_script.exists():
        print(f"Error: Ray tracer script not found at {ray_tracer_script}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find all .txt scene files
    scene_files = sorted(examples_dir.glob("*.txt"))

    if not scene_files:
        print(f"No .txt scene files found in {examples_dir}")
        sys.exit(1)

    print(f"Found {len(scene_files)} scene file(s) to render")
    print(f"Using {args.workers} parallel worker(s)\n")
    print("="*70)

    # Prepare rendering tasks
    tasks = []
    for scene_file in scene_files:
        scene_name = scene_file.stem
        output_image = output_dir / f"{scene_name}.png"
        tasks.append((scene_file, output_image))

    # Start timing
    start_time = time.time()

    # Render scenes in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_scene = {
            executor.submit(
                render_scene,
                scene_file,
                output_image,
                ray_tracer_script,
                args.width,
                args.height
            ): scene_file.stem
            for scene_file, output_image in tasks
        }

        # Process results as they complete
        for future in as_completed(future_to_scene):
            scene_name = future_to_scene[future]
            try:
                name, success, message = future.result()
                results.append((name, success))
                completed += 1

                print(f"\n{message}")

            except Exception as exc:
                print(f"\n✗ {scene_name} generated exception: {exc}")
                results.append((scene_name, False))
                completed += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    print("="*70)
    print("SUMMARY")
    print("="*70)

    # Count successes and failures
    successes = sum(1 for _, success in results if success)
    failures = len(results) - successes

    print(f"Total scenes:     {len(results)}")
    print(f"Successful:       {successes}")
    print(f"Failed:           {failures}")
    print(f"Elapsed time:     {elapsed_time:.2f} seconds")
    print(f"Average per scene: {elapsed_time/len(results):.2f} seconds")
    print(f"\nAll images saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
