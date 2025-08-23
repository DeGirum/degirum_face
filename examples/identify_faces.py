#!/usr/bin/env python3
"""
Identify faces in an image using the full pipeline.
Usage:
    python identify_faces.py <image_path>
"""
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import degirum_face from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from degirum_face import FaceRecognition
import degirum_tools
from degirum_face.face_filters import FaceFilter, FaceFilterConfig

if len(sys.argv) != 2:
    print("Usage: python identify_faces.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

from degirum_face.face_filters import FaceFilterConfig

# Set a threshold that will always reject faces (e.g., min_face_size very large)
reject_all_config = FaceFilterConfig(min_face_size=10)  # Unrealistically large
face_filter = FaceFilter(reject_all_config)
face_rec = FaceRecognition.auto("hailo8", enable_logging=False, face_filter=face_filter)

results_obj = face_rec.identify_faces(image_path)

detections = results_obj.results
print(detections)
if not detections:
    print("No faces found.")
else:
    for i, det in enumerate(detections, 1):
        label = det.get("label", "Unknown")
        similarity = det.get("similarity", 0.0)
        print(f"Face {i}: {label} (similarity: {similarity:.3f})")
with degirum_tools.Display("Display results") as display:
    display.show_image(results_obj.image_overlay)
