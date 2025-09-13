#
# face_recognition_simple.py: Simplest Face Recognition Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a simple face recognition example using DeGirum Face Tracking library.
# This example demonstrates how to set up a face recognition pipeline and run it on a set of images.
#
# You can configure all the settings in the `face_recognition.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
# - Run `face_recognition_enroll.py` example to populate the ReID database.
#

import degirum_face, sys, os


def main():
    # Check if any image paths were provided
    if len(sys.argv) < 2:
        print(
            f"Usage: python {os.path.basename(__file__)} <image_path1> [image_path2] [image_path3] ..."
        )
        sys.exit(1)

    # load settings from YAML file
    config, _ = degirum_face.FaceRecognitionConfig.from_yaml(
        yaml_file="face_recognition.yaml"
    )

    # create FaceRecognition instance
    face_recognition = degirum_face.FaceRecognition(config)

    # recognize faces iterating over command line arguments
    for result in face_recognition.recognize_batch(iter(sys.argv[1:])):
        print(f"\nResults for {result.info}:")
        for face in result.results:
            print(degirum_face.FaceRecognitionResult.from_dict(face))


if __name__ == "__main__":
    main()
