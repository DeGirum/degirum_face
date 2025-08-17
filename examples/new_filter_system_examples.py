"""
Example usage of the new ChatGPT-inspired filtering pipeline architecture.

This example demonstrates the improved modular filtering system with individual
filter classes, pipeline control, and better debugging capabilities.
"""

from degirum_face.face_tracking_gizmos import (
    # New filter classes
    DetectionFilter,
    LandmarksFilter,
    SizeFilter,
    ZoneFilter,
    FrontalFilter,
    ShiftFilter,
    FrameFilterPipeline,
    ReIDFilter,
    # Legacy classes (still supported)
    FaceFilter,
    FaceFilterConfig,
    FaceExtractGizmo,
    ObjectMap,
    FaceStatus,
)
import numpy as np


def example_individual_filters():
    """Example of using individual filter classes."""
    print("=== Individual Filter Examples ===")

    # Create sample face detection result
    face_result = {
        "bbox": [100, 100, 200, 200],
        "landmarks": [
            {"landmark": [120, 130]},  # left eye
            {"landmark": [180, 130]},  # right eye
            {"landmark": [150, 160]},  # nose
            {"landmark": [130, 190]},  # left mouth
            {"landmark": [170, 190]},  # right mouth
        ],
        "track_id": 1,
    }

    # Test individual filters
    landmarks_filter = LandmarksFilter()
    ok, reason = landmarks_filter(face_result)
    print(f"LandmarksFilter: {ok}, reason: {reason}")

    size_filter = SizeFilter(min_side=50)
    ok, reason = size_filter(face_result)
    print(f"SizeFilter: {ok}, reason: {reason}")

    # Test with small face
    small_face = face_result.copy()
    small_face["bbox"] = [100, 100, 120, 120]  # 20x20 face
    ok, reason = size_filter(small_face)
    print(f"SizeFilter (small): {ok}, reason: {reason}")

    zone_filter = ZoneFilter(zone_ids=[1, 2])
    # Mock zone data
    face_result["in_zone"] = [False, True, False, False]  # In zone 1
    ok, reason = zone_filter(face_result)
    print(f"ZoneFilter: {ok}, reason: {reason}")


def example_pipeline_system():
    """Example of using the new pipeline system."""
    print("\n=== Pipeline System Examples ===")

    # Create pipeline with custom configuration
    pipeline = FrameFilterPipeline(
        zone_ids=[1, 2, 3],
        min_face_size=64,
        enable_frontal_filter=True,
        enable_shift_filter=True,
    )

    # Sample face that should pass all filters
    good_face = {
        "bbox": [100, 100, 200, 200],  # Large enough
        "landmarks": [
            {"landmark": [120, 130]},
            {"landmark": [180, 130]},
            {"landmark": [150, 160]},
            {"landmark": [130, 190]},
            {"landmark": [170, 190]},
        ],
        "in_zone": [False, True, False, False],  # In zone 1
        "track_id": 1,
    }

    # Test pre-filters (landmarks, size, zone)
    ok, reason = pipeline.apply_pre(good_face)
    print(f"Pre-filters: {ok}, reason: {reason}")

    # Test post-filters (frontal, shift) - these will call static methods
    # For demo purposes, we'll skip the actual face analysis
    print("Post-filters would test frontal and shift detection")


def example_reid_filter():
    """Example of using the ReID filter for scheduling."""
    print("\n=== ReID Filter Examples ===")

    # Create object map and ReID filter
    object_map = ObjectMap[FaceStatus]()
    reid_filter = ReIDFilter(object_map, reid_expiration_frames=30)

    track_id = 1

    # First frame - should process
    should_process = reid_filter.should_reid(track_id, frame_id=100)
    print(f"Frame 100, Track {track_id}: {should_process}")

    # Same frame - should not process again
    should_process = reid_filter.should_reid(track_id, frame_id=100)
    print(f"Frame 100 (repeat), Track {track_id}: {should_process}")

    # Next frame - should not process yet (too soon)
    should_process = reid_filter.should_reid(track_id, frame_id=101)
    print(f"Frame 101, Track {track_id}: {should_process}")

    # Frame where it should process again
    should_process = reid_filter.should_reid(track_id, frame_id=102)
    print(f"Frame 102, Track {track_id}: {should_process}")


def example_custom_filter():
    """Example of creating a custom filter."""
    print("\n=== Custom Filter Example ===")

    class ConfidenceFilter(DetectionFilter):
        """Custom filter that checks detection confidence."""

        name: str = "confidence"

        def __init__(self, min_confidence: float = 0.5):
            self.min_confidence = min_confidence

        def __call__(self, det: dict) -> tuple:
            confidence = det.get("confidence", 0.0)
            ok = confidence >= self.min_confidence
            reason = (
                None
                if ok
                else f"confidence:low({confidence:.2f}<{self.min_confidence})"
            )
            return ok, reason

    # Use custom filter
    confidence_filter = ConfidenceFilter(min_confidence=0.7)

    high_conf_face = {"confidence": 0.9}
    ok, reason = confidence_filter(high_conf_face)
    print(f"High confidence face: {ok}, reason: {reason}")

    low_conf_face = {"confidence": 0.3}
    ok, reason = confidence_filter(low_conf_face)
    print(f"Low confidence face: {ok}, reason: {reason}")


def example_pipeline_with_custom_filters():
    """Example of extending the pipeline with custom filters."""
    print("\n=== Extended Pipeline Example ===")

    class CustomFrameFilterPipeline(FrameFilterPipeline):
        """Extended pipeline with additional filters."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Custom filters can be added here
            # Example: self._pre_filters.append(ConfidenceFilter(min_confidence=0.6))

    print("Custom pipeline can extend the base system with additional filters")


def example_backward_compatibility():
    """Example showing backward compatibility with legacy system."""
    print("\n=== Backward Compatibility Example ===")

    # Old way still works
    legacy_config = FaceFilterConfig(
        min_face_size=50, zone_ids=[1, 2], enable_frontal_filter=True
    )

    legacy_filter = FaceFilter(config=legacy_config)

    face_result = {
        "bbox": [100, 100, 200, 200],
        "landmarks": [{"landmark": [120, 130]} for _ in range(5)],
        "in_zone": [False, True, False],
    }

    should_process = legacy_filter.should_process_face(face_result)
    print(f"Legacy filter result: {should_process}")

    # New gizmo can still use old configuration
    gizmo = FaceExtractGizmo(target_image_size=112, filter_config=legacy_config)
    print("Legacy configuration works with new gizmo implementation")


def example_debugging_with_reasons():
    """Example showing improved debugging with filter reasons."""
    print("\n=== Debugging with Filter Reasons ===")

    # Test various failure scenarios
    test_cases = [
        {
            "name": "Missing landmarks",
            "face": {"bbox": [100, 100, 200, 200]},
        },
        {
            "name": "Too small",
            "face": {
                "bbox": [100, 100, 110, 110],  # 10x10 face
                "landmarks": [{"landmark": [105, 105]} for _ in range(5)],
            },
        },
        {
            "name": "Wrong zone",
            "face": {
                "bbox": [100, 100, 200, 200],
                "landmarks": [{"landmark": [120, 130]} for _ in range(5)],
                "in_zone": [True, False, False],  # In zone 0, not 1 or 2
            },
        },
    ]

    # Create filters
    landmarks_filter = LandmarksFilter()
    size_filter = SizeFilter(min_side=50)
    zone_filter = ZoneFilter(zone_ids=[1, 2])

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        face = test_case["face"]

        # Mock zone data if needed
        if "in_zone" not in face and zone_filter.zone_ids:
            face["in_zone"] = [False, False, False]

        ok, reason = landmarks_filter(face)
        print(f"  Landmarks: {ok}, reason: {reason}")

        ok, reason = size_filter(face)
        print(f"  Size: {ok}, reason: {reason}")

        ok, reason = zone_filter(face)
        print(f"  Zone: {ok}, reason: {reason}")


if __name__ == "__main__":
    print("New ChatGPT-Inspired Filter System Examples")
    print("=" * 50)

    example_individual_filters()
    example_pipeline_system()
    example_reid_filter()
    example_custom_filter()
    example_pipeline_with_custom_filters()
    example_backward_compatibility()
    example_debugging_with_reasons()

    print("\n" + "=" * 50)
    print("All examples completed!")
