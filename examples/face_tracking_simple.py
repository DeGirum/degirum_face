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
# so it is recommended to run that example first unless you explicitly want to collect clips of all
# discovered faces.
#
# When unknown face is detected, it saves the video clip to the configured storage
# and sends a notification.
#
# You can configure all the settings in the `face_tracking.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
# - Run `face_tracking_web_app.py` or `face_tracking_add_embeddings.py` examples to populate the ReID database.
#

from pathlib import Path
import degirum_face


def main():

    # load settings from YAML file
    config, _ = degirum_face.FaceTrackingConfig.from_yaml(
        yaml_file="face_tracking.yaml"
    )

    # set live stream mode to local window
    config.live_stream_mode = "LOCAL"

    # start face tracking pipeline
    composition, _ = degirum_face.start_face_tracking_pipeline(config)

    # wait for the composition to finish
    composition.wait()


if __name__ == "__main__":
    main()
