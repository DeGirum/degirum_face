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

if len(sys.argv) != 2:
    print("Usage: python identify_faces.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

face_rec = FaceRecognition.auto("hailo8", enable_logging=False)

results_obj = face_rec.identify_faces(image_path)

detections = results_obj.results
if not detections:
    print("No faces found.")
else:
    for i, det in enumerate(detections, 1):
        label = det.get("label", "Unknown")
        similarity = det.get("similarity", 0.0)
        print(f"Face {i}: {label} (similarity: {similarity:.3f})")
with degirum_tools.Display("Display results") as display:
    display.show_image(results_obj.image_overlay)
