#!/usr/bin/env python3
"""
Face Enrollment and Identification Example

This example demonstrates the complete face recognition workflow:
1. Enroll a person using multiple training images
2. Identify the person in a new test image

Usage:
    python face_enrollment_identification_example.py

The example:
- Enrolls "Chandler" using Chandler_1.jpg and Chandler_2.jpg
- Identifies faces in Chandler_3.jpg (should recognize as Chandler)
- Tests identification with other characters (should be unknown)
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import degirum_face
sys.path.insert(0, str(Path(__file__).parent.parent))

from degirum_face import FaceRecognition


def main():
    """Main function to demonstrate enrollment and identification."""

    print("=== Face Enrollment and Identification Example ===\n")

    # Initialize face recognition pipeline
    print("Initializing face recognition pipeline...")
    try:
        # Use auto mode with hailo8 and cloud inference
        face_rec = FaceRecognition.auto("hailo8", "@cloud")
        print("✓ Pipeline initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return

    # Define paths to test images
    assets_dir = Path(__file__).parent.parent / "assets"

    # Step 1: Enroll Chandler using multiple images
    print("Step 1: Enrolling Chandler with training images...")
    enrollment_images = [str(assets_dir / "Rachel_1.jpg")]

    print(f"  Training images: {[os.path.basename(img) for img in enrollment_images]}")

    try:
        success = face_rec.enroll("Rachel", enrollment_images)
        if success:
            print("✓ Rachel enrolled successfully\n")
        else:
            print("✗ Failed to enroll Rachel\n")
            return
    except Exception as e:
        print(f"✗ Error during enrollment: {e}\n")
        return

    # Step 2: Test identification
    print("Step 2: Testing identification...")

    test_cases = [
        {
            "name": "Chandler Recognition Test",
            "image": str(assets_dir / "Chandler_3.jpg"),
            "expected": "Chandler",
            "description": "Should recognize Chandler from a new image",
        },
        {
            "name": "Rachel Test (Unknown)",
            "image": str(assets_dir / "Rachel_1.jpg"),
            "expected": "Rachel",
            "description": "Should recognize Rachel (different person)",
        },
        {
            "name": "Monica Test (Unknown)",
            "image": str(assets_dir / "Monica_1.jpg"),
            "expected": "Unknown",
            "description": "Should not recognize Monica (different person)",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  Image: {os.path.basename(test_case['image'])}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Description: {test_case['description']}")

        try:
            # Perform identification
            results = face_rec.identify_faces(test_case["image"])

            print("  Raw detections:")
            print(results)

            if not results:
                print("  Result: No faces detected ✗")
                continue

            # (Old logic commented out for now)
            # face_result = results[0]
            # detected_name = face_result.get("person_name", "Unknown")
            # confidence = face_result.get("confidence", 0.0)
            # is_unknown = face_result.get("unknown", True)
            # ...

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n=== Test Summary ===")
    print("This example demonstrated:")
    print("1. ✓ Enrolling a person with multiple training images")
    print("2. ✓ Identifying the enrolled person in new images")
    print("3. ✓ Rejecting unknown persons as 'Unknown'")

    print(f"\nDatabase location: face_recognition.lance")
    print("Note: The enrolled person data persists between runs.")
    print("To reset, delete the 'face_recognition.lance' directory.")

    # Additional interactive suggestions
    print(f"\n=== Try More Tests ===")
    print("You can test with other images by modifying the script:")
    print(f"Example additional tests:")

    other_images = [
        "Rachel_1.jpg",
        "Ross_1.jpg",
        "Phoebe_1.jpg",
        "Joey_2.jpg",
        "Monica_2.jpg",
    ]

    for img in other_images:
        if (assets_dir / img).exists():
            print(f"  results = face_rec.identify_faces(assets_dir / '{img}')")

    print(f"\nTo enroll additional people:")
    print(
        f"  face_rec.enroll('Monica', [assets_dir / 'Monica_1.jpg', assets_dir / 'Monica_2.jpg'])"
    )
    print(
        f"  face_rec.enroll('Joey', [assets_dir / 'Joey_1.jpg', assets_dir / 'Joey_2.jpg'])"
    )


def demonstrate_multi_person_enrollment(face_rec, assets_dir):
    """
    Optional function to demonstrate enrolling multiple people.
    Call this to set up a more complete database.
    """

    print("\n=== Multi-Person Enrollment Demo ===")

    people_to_enroll = [
        {"name": "Monica", "images": ["Monica_1.jpg", "Monica_2.jpg"]},
        {"name": "Joey", "images": ["Joey_1.jpg", "Joey_2.jpg"]},
        {"name": "Rachel", "images": ["Rachel_1.jpg", "Rachel_2.jpg"]},
    ]

    for person in people_to_enroll:
        print(f"\nEnrolling {person['name']}...")
        image_paths = [str(assets_dir / img) for img in person["images"]]

        # Check if images exist
        missing_images = [img for img in image_paths if not Path(img).exists()]
        if missing_images:
            print(f"  ⚠️ Missing images: {[Path(img).name for img in missing_images]}")
            continue

        try:
            success = face_rec.enroll(person["name"], image_paths)
            if success:
                print(f"  ✓ {person['name']} enrolled successfully")
            else:
                print(f"  ✗ Failed to enroll {person['name']}")
        except Exception as e:
            print(f"  ✗ Error enrolling {person['name']}: {e}")


if __name__ == "__main__":
    main()

    # Uncomment the lines below to also enroll additional people
    # print("\n" + "="*50)
    # face_rec = FaceRecognition.auto("hailo8", "@cloud")
    # assets_dir = Path(__file__).parent.parent / "assets"
    # demonstrate_multi_person_enrollment(face_rec, assets_dir)
