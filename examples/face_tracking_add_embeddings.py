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
import degirum_face


def main():
    # load settings from YAML file
    config = degirum_face.FaceAnnotationConfig.from_yaml(yaml_file="face_tracking.yaml")

    # create FaceAnnotation instance
    annotator = degirum_face.FaceAnnotation(config)

    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python face_tracking_add_embeddings.py <clip_name> <person_name>")
        print("Example: python face_tracking_add_embeddings.py 0000123 'John Doe'")
        print("Available video clips in storage:")
        for clip_info in annotator.list_clips().values():
            if "original" in clip_info:
                print(f"  {clip_info['original'].object_name}")
        print("Person names currently in database:")
        for person in config.db.list_objects().values():
            print(f"  {person}")
        sys.exit(1)

    video_file = sys.argv[1]
    person_name = sys.argv[2]

    # run analysis pipeline on the video file
    print(f"Processing video file: {video_file}")
    face_map = annotator.run_clip_annotation(video_file, save_annotated=False)

    if len(face_map.map) != 1:
        print(
            f"Error: {len(face_map.map)} faces detected. This example assumes exactly one person."
        )
        sys.exit(1)

    face_obj = next(iter(face_map.map.values()))

    # add embeddings to the database
    cnt, obj_id = config.db.add_embeddings_for_attributes(
        person_name, face_obj.embeddings
    )
    print(
        f"Successfully added {cnt} embeddings for '{person_name}' (object ID = {obj_id}) to the database."
    )


if __name__ == "__main__":
    main()
