#
# face_tracking_add_embeddings: Face Tracking Example How to Add Embeddings to Database
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements analysis of provided video clip: face detection, embeddings computation, and
# adding embeddings to the ReID database.
# This example assumes that provided video clip contains exactly one person.
#
# Usage: `python face_tracking_add_embeddings.py <video_clip> <person_name>`
#
# When you run this example without arguments, it will list available video clips in the object storage
# and known persons in the database. <video_clip> should be among the listed video clips.
#
# You can configure all the settings in the `face_tracking.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
# - Run `face_tracking_web_app.py` or `face_tracking_simple.py` examples to collect video clips of unknown persons
#


import sys
import uuid
import yaml
import degirum_face
from pathlib import Path
from degirum_tools import ObjectStorageConfig, ObjectStorage


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

    # define clip storage configuration
    clip_storage_config = ObjectStorageConfig(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
    )

    # create FaceTracking instance
    face_tracker = degirum_face.FaceTracking(
        hw_location=hw_location,
        model_zoo_url=model_zoo_url,
        face_detector_model_name=face_detector_model_name,
        face_detector_model_devices=face_detector_model_devices,
        face_reid_model_name=face_reid_model_name,
        face_reid_model_devices=face_reid_model_devices,
        clip_storage_config=clip_storage_config,
        db_filename=db_filename,
    )

    # list clips available in the storage
    clips = face_tracker.list_clips()

    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python face_tracking_add_embeddings.py <clip_name> <person_name>")
        print("Example: python face_tracking_add_embeddings.py 0000123 'John Doe'")
        print("Available video clips in storage:")
        for clip_name in clips:
            print(f"  {clip_name}")
        print("Person names currently in database:")
        for person in face_tracker.db.list_objects().values():
            print(f"  {person}")
        sys.exit(1)

    video_file = sys.argv[1]
    person_name = sys.argv[2]

    # check if video file exists
    clip_name = Path(video_file).stem
    clip_dict = clips.get(clip_name, None)
    if clip_dict is None or "original" not in clip_dict:
        raise RuntimeError(
            f"Error: video file '{video_file}' not found in object storage {clip_storage_config}"
        )

    # run analysis pipeline on the video file
    print(f"Processing video file: {video_file}")
    face_map = face_tracker.run_clip_analysis(
        clip_dict["original"], zone, save_annotated=False
    )

    if len(face_map.map) != 1:
        print(
            f"Error: {len(face_map.map)} faces detected. This example assumes exactly one person."
        )
        sys.exit(1)

    for face_obj in face_map.map.values():
        obj_id = face_tracker.db.get_id_by_attributes(person_name)
        if obj_id is None:
            # add the person to the database
            obj_id = str(uuid.uuid4())
            face_tracker.db.add_object(obj_id, person_name)
            print(
                f"NOTE: added new person '{person_name}' with object ID = {obj_id} to the database."
            )

        # add embeddings to the database
        face_tracker.db.add_embeddings(obj_id, face_obj.embeddings)
        break

    print(
        f"Successfully added {len(face_obj.embeddings)} embeddings for '{person_name}' (object ID = {obj_id}) to the database."
    )


if __name__ == "__main__":
    main()
