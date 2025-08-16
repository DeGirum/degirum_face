#
# face_tracking_simple.py: Simplest Face Tracking Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a simple face tracking example using DeGirum Face Tracking library.
# This example demonstrates how to set up a face tracking pipeline and display live video on local UI.
# It uses the same YAML setup as the `face_tracking_web_app.py` example.
# It uses ReID database which is filled with embeddings from the `face_tracking_web_app.py` example,
# so it is recommended to run that example first.
#
# When unknown face is detected, it saves the video clip to the configured storage
# and sends a notification.
#
# You can configure all the settings in the `face_tracking.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
# - Run `face_tracking_web_app.py` example to populate the ReID database.
#


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
video_source = 0
db_filename = "face_reid_db.lance"
zone = None
clip_duration = 100
reid_expiration_frames = 10
notification_config = notification_config_console
notification_message = (
    "{time}: Unknown person detected (saved video: [{filename}]({url})])"
)
credence_count = 4
endpoint = "./"
access_key = ""
secret_key = ""
bucket = "unknown_faces"


def main():
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
        clip_storage_config=ObjectStorageConfig(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
        ),
        db_filename=db_filename,
    )

    # run the face tracking pipeline
    composition, _ = face_tracker.run_tracking_pipeline(
        video_source,
        zone=zone,
        clip_duration=clip_duration,
        reid_expiration_frames=reid_expiration_frames,
        credence_count=credence_count,
        alert_mode=degirum_face.AlertMode.ON_UNKNOWNS,
        alert_once=True,
        notification_config=notification_config,
        notification_message=notification_message,
        local_display=True,
        stream_name="Face Tracking Example, press 'q' to exit",
    )

    # wait for the composition to finish
    composition.wait()


if __name__ == "__main__":
    main()
