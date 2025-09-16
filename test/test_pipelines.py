#
# test_pipelines.py: Integration tests for degirum-face package pipelines
# Copyright DeGirum Corp. 2025
#
# Integration tests that mirror the steps performed in Tutorials.ipynb
# Tests both face recognition and face tracking functionality end-to-end
#

import pytest, os
import degirum, degirum_face, degirum_tools


# Helper functions for setup
def create_face_recognition_setup(temp_dir, assets_dir):
    """Create face recognition components as in tutorial"""
    # Use cloud hardware for integration tests
    hardware_to_use = "N2X/ORCA1"

    # Create face embeddings database
    db_path = os.path.join(temp_dir, "test_tutorial_db.lance")
    db = degirum_face.ReID_Database(db_path)

    # Define model specs (using cloud models for integration tests)
    face_detection_model = (
        degirum_face.model_registry.for_task("face_detection")
        .for_hardware(hardware_to_use)
        .top_model_spec()
    )
    face_embedding_model = (
        degirum_face.model_registry.for_task("face_embedding")
        .for_hardware(hardware_to_use)
        .top_model_spec()
    )

    # Create face recognition configuration
    face_recognition_config = degirum_face.FaceRecognitionConfig(
        face_detection_model=face_detection_model,
        face_embedding_model=face_embedding_model,
        db=db,
    )

    # Create FaceRecognition instance
    face_recognition = degirum_face.FaceRecognition(face_recognition_config)

    return {
        "db": db,
        "face_recognition": face_recognition,
    }


def create_face_tracking_setup(temp_dir, assets_dir):
    """Create face tracking components as in tutorial"""
    hardware_to_use = "N2X/ORCA1"

    # Get video path
    video_path = os.path.join(assets_dir, "WalkingPeople-short.mp4")
    if not os.path.exists(video_path):
        pytest.skip(f"Video asset not found: {video_path}")

    # Create face embeddings database
    db_path = os.path.join(temp_dir, "test_tracking_db.lance")
    db = degirum_face.ReID_Database(db_path)

    # Define model specs
    face_detection_model = (
        degirum_face.model_registry.for_task("face_detection")
        .for_hardware(hardware_to_use)
        .top_model_spec()
    )
    face_embedding_model = (
        degirum_face.model_registry.for_task("face_embedding")
        .for_hardware(hardware_to_use)
        .top_model_spec()
    )

    # Define face filter configuration
    face_filter_config = degirum_face.FaceFilterConfig(
        enable_small_face_filter=True,
        min_face_size=30,
        enable_zone_filter=True,
        zone=[[100, 10], [960, 10], [960, 700], [100, 700]],
        enable_frontal_filter=True,
        enable_shift_filter=True,
        enable_reid_expiration_filter=True,
        reid_expiration_frames=4,
    )

    # Define clip storage configuration
    clip_storage_config = degirum_tools.ObjectStorageConfig(
        endpoint=temp_dir,
        access_key="",
        secret_key="",
        bucket="test_videos",
    )

    # Define face tracking configuration
    face_tracking_config = degirum_face.FaceTrackingConfig(
        video_source=video_path,
        face_detection_model=face_detection_model,
        face_embedding_model=face_embedding_model,
        db=db,
        face_filter_config=face_filter_config,
        clip_storage_config=clip_storage_config,
        clip_duration=20,
        alert_mode=degirum_face.AlertMode.ON_UNKNOWNS,
        credence_count=2,
        notification_timeout_s=20.0,
        live_stream_mode="NONE",
    )

    # Create FaceAnnotation instance
    face_annotation = degirum_face.FaceAnnotation(face_tracking_config)

    return {
        "face_tracking_config": face_tracking_config,
        "face_annotation": face_annotation,
        "db": db,
    }


def test_face_recognition(temp_dir, assets_dir):
    """Test complete face recognition pipeline: database enrollment and batch recognition"""

    setup = create_face_recognition_setup(temp_dir, assets_dir)
    db = setup["db"]
    face_recognition = setup["face_recognition"]

    #
    # Step 1: Clear database and enroll Alice and Bob
    #

    db.clear_all_tables()

    # Verify database is empty
    assert len(db.count_embeddings()) == 0

    # Enroll Alice and Bob
    alice_image = os.path.join(assets_dir, "Alice-1.png")
    bob_image = os.path.join(assets_dir, "Bob-1.png")
    enrolled = face_recognition.enroll_batch((alice_image, bob_image), ("Alice", "Bob"))

    # Verify enrollment was successful
    assert enrolled is not None
    embeddings_count = db.count_embeddings()
    assert len(embeddings_count) > 0, "No embeddings were stored in database"

    # Verify that we have exactly 2 objects (Alice and Bob)
    assert (
        len(embeddings_count) == 2
    ), f"Expected 2 objects, but got {len(embeddings_count)}"

    # Verify each object has exactly 1 embedding
    enrolled_names = set()
    for object_id, (count, attributes) in embeddings_count.items():
        assert (
            count == 1
        ), f"Expected 1 embedding for object {object_id}, but got {count}"
        enrolled_names.add(attributes)

    # Verify we have Alice and Bob enrolled
    assert "Alice" in enrolled_names, "Alice was not found in enrolled objects"
    assert "Bob" in enrolled_names, "Bob was not found in enrolled objects"

    #
    # Step 2: Recognize enrolled persons using other images
    #
    test_images = [
        "Alice-2.png",
        "Alice-3.png",
        "Bob-2.png",
        "Bob-3.png",
        "Alice&Bob.png",
    ]

    # Verify all test images exist
    image_paths = []
    for img in test_images:
        img_path = os.path.join(assets_dir, img)
        assert os.path.exists(img_path), f"Test image not found: {img_path}"
        image_paths.append(img_path)

    # Run recognition on batch of images
    results_count = 0
    for result in face_recognition.recognize_batch(tuple(image_paths)):
        assert isinstance(result, degirum.postprocessor.DetectionResults)

        for face in result.results:
            face_result = degirum_face.FaceRecognitionResult.from_dict(face)

            # Verify face result has required attributes
            assert face_result.bbox is not None and face_result.bbox == face.get("bbox")
            assert (
                face_result.detection_score is not None
                and face_result.detection_score == face.get("score")
            )
            assert (
                face_result.similarity_score is not None
                and face_result.similarity_score == face.get("face_similarity_score")
            )
            assert (
                face_result.landmarks is not None
                and face_result.landmarks == face.get("landmarks")
            )
            assert (
                face_result.attributes is not None
                and face_result.attributes == face.get("face_attributes")
            )
            assert face_result.db_id is not None and face_result.db_id == face.get(
                "face_db_id"
            )
            assert (
                face_result.embeddings is not None
                and len(face_result.embeddings) == 1
                and all(face_result.embeddings[0] == face.get("face_embeddings"))
            )

            assert face_result.attributes is not None, "Face result has no attributes"
            assert face_result.attributes in {"Alice", "Bob"}, "Unexpected attributes"
            results_count += 1

    # Verify we got all recognition results
    assert results_count == 6, "Not all faces were recognized"


# Face Tracking Pipeline Tests
def test_face_tracking_pipeline_execution(temp_dir, assets_dir):
    """Test for face tracking pipeline, collect clips, annotate clips"""

    #
    # Test Step 1: Run face tracking pipeline and collect clips
    #
    setup = create_face_tracking_setup(temp_dir, assets_dir)
    face_annotation = setup["face_annotation"]
    face_tracking_config = setup["face_tracking_config"]
    db = setup["db"]

    # Clear any existing clips
    face_annotation.remove_all_clips()

    # Verify no clips exist initially
    initial_clips = face_annotation.list_clips()
    assert len(initial_clips) == 0, "Clips should be empty after removal"

    # Run the face tracking pipeline
    composition, _ = degirum_face.start_face_tracking_pipeline(face_tracking_config)
    composition.wait()

    # Check if clips were generated
    clips = face_annotation.list_clips()
    assert len(clips) == 1, "Clips should be generated after pipeline run"

    #
    # Test Step 2: Analyze and annotate collected clips
    #

    # Test clip download and annotation for first clip
    first_clip = next(iter(clips))

    # Download original clip
    clip_data = face_annotation.download_clip(first_clip + ".mp4")
    assert clip_data is not None and len(clip_data) > 0

    # Annotate clip
    face_map = face_annotation.run_clip_annotation(first_clip)
    assert len(face_map.map) == 2, "Expected two tracked objects in face map"

    # Download annotated clip
    annotated_clip = face_annotation.download_clip(
        first_clip + face_annotation.annotated_video_suffix
    )
    assert annotated_clip is not None and len(annotated_clip) > 0

    #
    # Test Step 3: Enroll detected persons to database
    #

    attributes = []
    for track_id, tracked_object in face_map.map.items():
        attributes.append(f"TestPerson{track_id}")
        db.add_embeddings_for_attributes(
            f"TestPerson{track_id}", tracked_object.embeddings
        )

    # validate that database now has two enrolled objects
    embedding_counts = db.count_embeddings()
    assert len(embedding_counts) == len(
        face_map.map
    ), "Expected two enrolled objects in database"
    assert all(ec[0] > 0 and ec[1] in attributes for ec in embedding_counts.values())

    #
    # Test Step 4: Run pipeline again and verify no alerts after enrollment
    #

    # Clear clips before second run
    face_annotation.remove_all_clips()

    # Run pipeline again
    composition, _ = degirum_face.start_face_tracking_pipeline(face_tracking_config)
    composition.wait()

    # Check for clips (should be fewer or none if recognition is working)
    clips_after_enrollment = face_annotation.list_clips()
    assert (
        len(clips_after_enrollment) == 0
    ), "No clips should be generated after enrollment"
