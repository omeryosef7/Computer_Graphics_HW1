#!/usr/bin/env python3
"""
Compare two images pixel by pixel to verify ray tracer output.
Usage: python compare_images.py [options] [<reference_image> <rendered_image>]
"""

import argparse
import sys
import numpy as np
from PIL import Image
from pathlib import Path


def compare_images(reference_path, rendered_path, tolerance=10):
    """
    Compare two images pixel by pixel.

    Args:
        reference_path: Path to the reference image
        rendered_path: Path to the rendered image
        tolerance: Maximum allowed difference per pixel channel (0-255)
                  Default is 10 to account for random sampling in shadow rays

    Returns:
        bool: True if images match within tolerance
    """
    # Load images
    try:
        ref_img = Image.open(reference_path)
        render_img = Image.open(rendered_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # Convert to RGB if needed
    if ref_img.mode != 'RGB':
        ref_img = ref_img.convert('RGB')
    if render_img.mode != 'RGB':
        render_img = render_img.convert('RGB')

    # Check dimensions
    if ref_img.size != render_img.size:
        print(f"❌ Images have different dimensions:")
        print(f"   Reference: {ref_img.size}")
        print(f"   Rendered:  {render_img.size}")
        return False

    width, height = ref_img.size
    print(f"Comparing images of size {width}x{height}...")

    # Convert to numpy arrays
    ref_array = np.array(ref_img)
    render_array = np.array(render_img)

    # Calculate pixel differences
    diff = np.abs(ref_array.astype(int) - render_array.astype(int))

    # Count pixels that differ beyond tolerance
    pixels_differ = np.any(diff > tolerance, axis=2)
    num_different = np.sum(pixels_differ)
    total_pixels = width * height

    # Calculate statistics
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)

    print(f"\nResults:")
    print(f"  Total pixels:        {total_pixels:,}")
    print(f"  Different pixels:    {num_different:,}")
    print(f"  Matching pixels:     {total_pixels - num_different:,}")
    print(f"  Match percentage:    {100 * (1 - num_different/total_pixels):.2f}%")
    print(f"  Max difference:      {max_diff}")
    print(f"  Average difference:  {avg_diff:.2f}")

    if num_different == 0:
        print(f"\n✓ Images are IDENTICAL!")
        return True
    else:
        print(f"\n✗ Images are DIFFERENT (tolerance: {tolerance})")

        # Show where differences are concentrated
        if num_different > 0:
            diff_positions = np.where(pixels_differ)
            print(f"\nDifference locations (first 10):")
            for i in range(min(10, len(diff_positions[0]))):
                y, x = diff_positions[0][i], diff_positions[1][i]
                ref_pixel = ref_array[y, x]
                render_pixel = render_array[y, x]
                print(f"  Pixel ({x}, {y}): ref={ref_pixel} render={render_pixel}")

        return False


def compare_all_examples(examples_dir, rendered_dir, tolerance=10):
    """
    Compare all rendered images with their reference counterparts.
    """
    examples_path = Path(examples_dir)
    rendered_path = Path(rendered_dir)

    if not examples_path.exists():
        print(f"Error: Examples directory not found: {examples_path}")
        return

    if not rendered_path.exists():
        print(f"Error: Rendered directory not found: {rendered_path}")
        return

    # Find all PNG files in examples
    reference_images = sorted(examples_path.glob("*.png"))

    if not reference_images:
        print(f"No reference PNG files found in {examples_path}")
        return

    print(f"Found {len(reference_images)} reference image(s)\n")
    print("="*70)

    results = []
    for ref_img in reference_images:
        rendered_img = rendered_path / ref_img.name

        print(f"\nComparing: {ref_img.name}")
        print("-"*70)

        if not rendered_img.exists():
            print(f"❌ Rendered image not found: {rendered_img}")
            results.append((ref_img.name, False))
            continue

        match = compare_images(ref_img, rendered_img, tolerance)
        results.append((ref_img.name, match))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, match in results:
        status = "✓ MATCH" if match else "✗ DIFFER"
        print(f"  {status:10} - {name}")

    matches = sum(1 for _, m in results if m)
    print(f"\nTotal: {matches}/{len(results)} images match")


def main():
    parser = argparse.ArgumentParser(
        description='Compare ray traced images with reference images'
    )
    parser.add_argument(
        'images',
        nargs='*',
        help='Two image paths to compare (reference and rendered). If omitted, compares all examples.'
    )
    parser.add_argument(
        '--tolerance',
        type=int,
        default=10,
        help='Maximum allowed pixel difference (0-255). Default: 10 (accounts for random shadow sampling)'
    )
    args = parser.parse_args()

    if len(args.images) == 0:
        # No arguments - compare all examples
        base_dir = Path(__file__).parent
        examples_dir = base_dir / "raytracer" / "Examples"
        rendered_dir = base_dir / "raytracer" / "Rendered"

        print(f"Comparing all rendered images with reference images (tolerance: {args.tolerance})...\n")
        compare_all_examples(examples_dir, rendered_dir, tolerance=args.tolerance)

    elif len(args.images) == 2:
        # Two arguments - compare specific images
        reference_path = args.images[0]
        rendered_path = args.images[1]

        print(f"Using tolerance: {args.tolerance}\n")
        compare_images(reference_path, rendered_path, tolerance=args.tolerance)

    else:
        print("Error: Provide either 0 arguments (compare all) or 2 arguments (compare specific images)")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
