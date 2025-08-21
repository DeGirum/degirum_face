#!/usr/bin/env python3
"""
Face Verification Example

This example demonstrates how to use the FaceRecognition.verify() method
to check if two images contain the same person.

Usage:
    python face_verification_example.py

The example tests various scenarios:
1. Same person, different photos (should return True)
2. Different people (should return False)
3. Edge cases and error handling
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import degirum_face
sys.path.insert(0, str(Path(__file__).parent.parent))

from degirum_face import FaceRecognition


def main():
    """Main function to demonstrate face verification."""

    print("=== Face Verification Example ===\n")

    # Initialize face recognition pipeline
    print("Initializing face recognition pipeline...")
    try:
        # Use auto mode with CPU inference (change hardware as needed)
        face_rec = FaceRecognition.auto("hailo8", "@cloud")
        print("✓ Pipeline initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return

    # Define paths to test images
    assets_dir = Path(__file__).parent.parent / "assets"

    # Test scenarios
    test_cases = [
        {
            "name": "Same Person - Chandler (different photos)",
            "image1": assets_dir / "Chandler_1.jpg",
            "image2": assets_dir / "Chandler_2.jpg",
            "expected": "MATCH",
        },
        {
            "name": "Same Person - Monica (different photos)",
            "image1": assets_dir / "Monica_1.jpg",
            "image2": assets_dir / "Monica_3.jpg",
            "expected": "MATCH",
        },
        {
            "name": "Different People - Chandler vs Joey",
            "image1": assets_dir / "Chandler_1.jpg",
            "image2": assets_dir / "Joey_1.jpg",
            "expected": "NO MATCH",
        },
        {
            "name": "Different People - Rachel vs Phoebe",
            "image1": assets_dir / "Rachel_1.jpg",
            "image2": assets_dir / "Phoebe_1.jpg",
            "expected": "NO MATCH",
        },
        {
            "name": "Same Person - Ross (different photos)",
            "image1": assets_dir / "Ross_1.jpg",
            "image2": assets_dir / "Ross_2.jpg",
            "expected": "MATCH",
        },
    ]

    # Run verification tests
    print("Running verification tests...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Image 1: {test_case['image1'].name}")
        print(f"  Image 2: {test_case['image2'].name}")
        print(f"  Expected: {test_case['expected']}")

        try:
            # Perform verification
            is_match, confidence = face_rec.verify(
                test_case["image1"], test_case["image2"]
            )

            # Display results
            result = "MATCH" if is_match else "NO MATCH"
            status = "✓" if result == test_case["expected"] else "✗"

            print(f"  Result: {result} (confidence: {confidence:.3f}) {status}")

            if result != test_case["expected"]:
                print(f"  ⚠️  Expected {test_case['expected']}, got {result}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()  # Empty line for readability

    print("=== Verification Tests Complete ===")

    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("You can now test your own image pairs!")
    print("Available images:")

    # List available images
    image_files = sorted([f for f in assets_dir.glob("*.jpg")])
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i:2d}. {img_file.name}")

    print("\nTo test verification manually, modify the paths below:")
    print("Example:")
    print("  image1 = assets_dir / 'Chandler_1.jpg'")
    print("  image2 = assets_dir / 'Chandler_3.jpg'")
    print("  is_match, confidence = face_rec.verify(image1, image2)")
    print(f"  print(f'Match: {{is_match}}, Confidence: {{confidence:.3f}}')")


def test_custom_images(face_rec, image1_path, image2_path):
    """
    Helper function to test verification on custom image paths.

    Args:
        face_rec: FaceRecognition instance
        image1_path: Path to first image
        image2_path: Path to second image
    """
    print(f"\nTesting verification:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")

    try:
        is_match, confidence = face_rec.verify(image1_path, image2_path)
        result = "MATCH" if is_match else "NO MATCH"
        print(f"  Result: {result} (confidence: {confidence:.3f})")
        return is_match, confidence
    except Exception as e:
        print(f"  Error: {e}")
        return False, 0.0


if __name__ == "__main__":
    main()
