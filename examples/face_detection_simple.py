#
# face_detection_simple.py: Simplest Face Detection Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a simple face detection example using DeGirum Face Tracking library.
# This example demonstrates how to set up a face detection pipeline and run it on a set of images.
#
# You can configure all the settings in the `face_tracking.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
# - Run `face_tracking_web_app.py` example to populate the ReID database.
#


import sys
import os
import yaml
import degirum_face
from degirum_tools import ObjectStorageConfig
from degirum_tools.streams import notification_config_console


#
# Setting (see `face_tracking.yaml` for detailed comments):
#
hw_location = "localhost"
model_zoo_url = "degirum/public"
face_detector_model_name = "yolov8n_relu6_widerface_kpts--640x640_quant_n2x_orca1_1"
face_detector_model_devices = None
face_reid_model_name = "mbf_w600k--112x112_float_n2x_orca1_1"
face_reid_model_devices = None
db_filename = "face_reid_db.lance"
zone = None


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

    # load YAML settings from file and update existing globals
    try:
        with open("face_tracking.yaml", "r") as f:
            settings = yaml.safe_load(f)
        globals().update({k: v for k, v in settings.items() if k in globals()})
    except Exception as e:
        print(f"ERROR loading settings from YAML: {str(e)}, using default values.")

    # create FaceTracking instance
    face_tracker = degirum_face.FaceTracking(
        hw_location=hw_location,
        model_zoo_url=model_zoo_url,
        face_detector_model_name=face_detector_model_name,
        face_detector_model_devices=face_detector_model_devices,
        face_reid_model_name=face_reid_model_name,
        face_reid_model_devices=face_reid_model_devices,
        db_filename=db_filename,
    )

    def image_paths_generator():
        """
        Generator function that yields command line parameters (image file paths).
        """
        for image_path in sys.argv[1:]:
            if os.path.isfile(image_path):
                yield image_path
            else:
                print(f"Warning: file '{image_path}' not found, skipping...")

    # process and display results
    for result in face_tracker.recognize_batch(image_paths_generator()):
        print(f"\nResults for {result.info}:")
        for i, face in enumerate(result.results):
            print(f"  Object #{i} ---")
            print(f"    Face Attributes : {face['face_attributes']}, ")
            print(f"    Similarity score: {face['face_similarity_score']:.2f}")
            print(f"    Bounding box    : [{', '.join(f'{x:.1f}' for x in face['bbox'])}]")
            print(f"    Detection score: {face['score']:.2f}")


if __name__ == "__main__":
    main()
