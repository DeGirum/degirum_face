import re
import sys
import os
import numpy as np
import degirum_tools

# Ensure local degirum_face is imported, not installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from degirum_face import FaceRecognition

# Example usage: python test_low_level_pipeline.py <image_path>


def main(image_path):
    # Initialize FaceRecognition in auto mode (adjust hardware as needed)
    face_rec = FaceRecognition.auto("hailo8")
    # Debug: Print enrolled persons and number of embeddings per person
    db = face_rec.db
    print("\n--- Database Debug Info ---")
    objects = db.list_objects()
    if not objects:
        print("No persons enrolled in DB.")
    else:
        print(f"Enrolled persons in DB: {list(objects.values())}")
        counts = db.count_embeddings()
        for obj_id, (count, attrs) in counts.items():
            print(f"Person '{attrs}' (object_id={obj_id}): {count} embeddings")
    print("--------------------------\n")

    # Step 1: Detect faces and get full results object
    results_obj = face_rec.detector.detect(image_path)  # PySDK results object
    detections = results_obj.results
    print(f"Detected {len(detections)} faces.")
    if not detections:
        print("No faces detected.")
        return

    # Step 2: Align faces (use results_obj.image)
    aligned_faces = face_rec.align_faces(results_obj.image, detections)
    print(f"Aligned {len(aligned_faces)} faces.")
    if not aligned_faces:
        print("No faces aligned.")
        return

    # Step 3: Get embeddings
    embeddings = face_rec.get_face_embeddings(aligned_faces)
    print(f"Got {len(embeddings)} embeddings.")
    if not embeddings:
        print("No embeddings generated.")
        return

    # Debug: Print test image embedding and Rachel's enrolled embedding, and their similarity
    print("\n--- Embedding Debug Info ---")
    print("Test image embedding:")
    print(embeddings[0])
    # Try to find Rachel's object_id
    rachel_id = None
    for obj_id, attrs in objects.items():
        if str(attrs).lower() == "rachel":
            rachel_id = obj_id
            break
    if rachel_id:
        rachel_embs = db.get_embeddings_by_id(rachel_id)
        if rachel_embs:
            print("Rachel's enrolled embedding:")
            print(rachel_embs[0])
            # Compute similarity
            sim = float(np.dot(embeddings[0], rachel_embs[0]))
            print(f"Similarity (test vs Rachel): {sim:.6f}")
        else:
            print("No embeddings found for Rachel in DB.")
    else:
        print("Rachel not found in DB.")
    print("----------------------------\n")

    # Note: If you observe negative similarity values very close to zero, this is likely due to floating point precision errors in dot product/cosine similarity. You may want to clamp similarity to [0, 1] if needed for your application.

    # Step 4: Identify faces (patches detections in-place)
    results = face_rec.get_identities(detections, embeddings)
    print(results_obj)
    with degirum_tools.Display("Display results") as display:
        display.show_image(results_obj.image_overlay)
    for i, det in enumerate(results):
        print(
            f"Face {i+1}: label={det.get('label')}, similarity={det.get('similarity'):.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_low_level_pipeline.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
