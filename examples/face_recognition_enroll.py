#
# face_recognition_enroll.py: Face Enrolling  Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a face recognition enrolling example using DeGirum Face Tracking library.
# This example demonstrates how to add face embeddings to the ReID database from images.
#
# You can configure all the settings in the `face_recognition.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
#

import sys
import degirum_face


def main():
    # load settings from YAML file
    config, _ = degirum_face.FaceRecognitionConfig.from_yaml(
        yaml_file="face_recognition.yaml"
    )

    # create FaceRecognition instance
    face_recognition = degirum_face.FaceRecognition(config)

    if len(sys.argv) >= 2 and sys.argv[1].lower() == "clear":
        # clear the database
        face_recognition.config.db.clear_all_tables()
        print("Database cleared")

    elif len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print(
            "Usage: python face_recognition_enroll.py <image_path1> <person_name1> [image_path2] [person_name2] ...\n"
            "To clear the database, run: python face_recognition_enroll.py clear"
        )
        print("Current number of embeddings in the database: ")
        print(face_recognition.config.db.count_embeddings())

    else:
        # recognize faces iterating over command line arguments
        embeddings = face_recognition.enroll_batch(
            iter(sys.argv[1::2]), iter(sys.argv[2::2])
        )
        print(f"Enrolled {len(embeddings)} faces")


if __name__ == "__main__":
    main()
