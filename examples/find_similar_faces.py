#
# find_similar_faces.py: Face Similarity Search Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a face similarity search example using DeGirum Face Tracking library.
# This example demonstrates how to find all photos with similar faces to a given input image.
#
# You can configure all the settings in the `face_recognition.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
#

import degirum_face, sys, os, glob


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: python {os.path.basename(__file__)} <reference_image_path> <photos_dir_wildcard>"
        )
        return

    # load settings from YAML file
    config, _ = degirum_face.FaceRecognitionConfig.from_yaml(
        yaml_file="face_recognition.yaml"
    )

    # create FaceRecognition instance
    face_recognition = degirum_face.FaceRecognition(config)

    reference_image_path = sys.argv[1]
    photos_directory = sys.argv[2]

    # clear existing data in the database
    config.db.clear_all_tables()

    # enroll a person from the reference image (biggest face in the image is used)
    person_name = "Reference"  # can be any string
    face_recognition.enroll_image(reference_image_path, person_name)

    # iterate over all image files in the directory (supports wildcards, e.g., *.jpg)
    image_files = glob.glob(photos_directory)

    # find and print all photos with similar faces to the reference image
    print(f"Photos with faces similar to {reference_image_path}:")
    for result in face_recognition.recognize_batch(image_files):
        for r in result.results:
            face = degirum_face.FaceRecognitionResult.from_dict(r)
            if face.attributes == person_name:
                print(f"{result.info}: similarity {face.similarity_score:.2f}")


if __name__ == "__main__":
    main()
