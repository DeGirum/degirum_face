#
# face_detection_simple.py: Simplest Face Recognition Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a simple face recognition example using DeGirum Face Tracking library.
# This example demonstrates how to set up a face recognition pipeline and run it on a set of images.
#
# You can configure all the settings in the `face_tracking.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
# - Run `face_tracking_web_app.py` example to populate the ReID database.
#

from pathlib import Path
import sys
import degirum_face


def main():
    # Check if any image paths were provided
    if len(sys.argv) < 2:
        print(
            "Usage: python face_detection_simple.py <image_path1> [image_path2] [image_path3] ..."
        )
        print(
            "Example: python face_detection_simple.py image1.jpg image2.png image3.jpg"
        )
        sys.exit(1)

    # load settings from YAML file
    config = degirum_face.FaceRecognitionConfig.from_yaml(
        yaml_file="face_tracking.yaml"
    )

    # create FaceRecognition instance
    face_recognition = degirum_face.FaceRecognition(config)

    for image_path in sys.argv[1:]:

        # recognize faces
        result = face_recognition.recognize_image(image_path)

        # display results
        print(f"\nResults for {result.info}:")
        for i, face in enumerate(result.results):
            print(f"  Object #{i} ---")
            print(f"    Face Attributes : {face['face_attributes']}, ")
            print(f"    Similarity score: {face['face_similarity_score']:.2f}")
            print(
                f"    Bounding box    : [{', '.join(f'{x:.1f}' for x in face['bbox'])}]"
            )
            print(f"    Detection score: {face['score']:.2f}")


if __name__ == "__main__":
    main()
