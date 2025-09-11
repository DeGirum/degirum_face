#
# group_similar_faces.py: Face Similarity Grouping Example
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements a face similarity grouping example using DeGirum Face Tracking library.
# This example demonstrates how to find groups of photos featuring similar faces.
#
# You can configure all the settings in the `face_recognition.yaml` file.
#
# Pre-requisites:
# - Install DeGirum Face SDK: `pip install degirum-face`
#

import degirum_face, sys, os, glob, sklearn, numpy as np


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <photos_dir_wildcard>")
        return

    photos = glob.glob(sys.argv[1])

    # load settings from YAML file
    config, _ = degirum_face.FaceRecognitionConfig.from_yaml(
        yaml_file="face_recognition.yaml"
    )

    # create FaceRecognition instance
    face_recognition = degirum_face.FaceRecognition(config)

    # compute face embeddings for all photos and store them together with filenames
    recognition_results: list = []
    for result in face_recognition.recognize_batch(photos):
        recognition_results.extend(
            (embedding, result.info)
            for r in result.results
            if (embedding := r.get("face_embeddings")) is not None
        )

    if len(recognition_results) == 0:
        print("No faces found")
        return

    # cluster embeddings using HDBSCAN
    clusterer = sklearn.cluster.HDBSCAN(min_cluster_size=2, metric="cosine")
    clusters = clusterer.fit_predict([e[0] for e in recognition_results])
    cluster_indexes = clusters.argsort()

    # print file groups
    current_cluster = -1
    for cluster, filename in zip(
        clusters[cluster_indexes], [recognition_results[i][1] for i in cluster_indexes]
    ):
        if cluster != -1:
            if cluster != current_cluster:
                print(f"\nPerson #{cluster} appears in:")
                current_cluster = cluster
            print(filename)


if __name__ == "__main__":
    main()
