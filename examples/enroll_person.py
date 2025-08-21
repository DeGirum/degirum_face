#!/usr/bin/env python3
"""
Enroll a person in the face database using a list of images.
Usage:
    python enroll_person.py <person_name> <image1> <image2> ...
"""
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import degirum_face from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from degirum_face import FaceRecognition

if len(sys.argv) < 3:
    print("Usage: python enroll_person.py <person_name> <image1> <image2> ...")
    sys.exit(1)

person_name = sys.argv[1]
image_paths = sys.argv[2:]

face_rec = FaceRecognition.auto("hailo8", enable_logging=False)

success = face_rec.enroll(person_name, image_paths)
if success:
    print(f"Enrolled '{person_name}' with {len(image_paths)} images.")
else:
    print(f"Failed to enroll '{person_name}'.")
