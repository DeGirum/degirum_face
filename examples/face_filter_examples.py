"""
Example usage of the refactored FaceExtractGizmo with standalone FaceFilter.

This example demonstrates how to use the new filtering architecture for better
reusability and modularity.
"""

from degirum_face.face_tracking_gizmos import (
    FaceFilter,
    FaceFilterConfig,
    FaceExtractGizmo,
    ObjectMap,
)
from degirum_face.face_data import FaceStatus


# Example 1: Using FaceFilter independently
def example_standalone_filter():
    """Example of using FaceFilter as a standalone component."""

    # Create filter configuration
    filter_config = FaceFilterConfig(
        min_face_size=50,
        zone_ids=[1, 2, 3],
        enable_frontal_filter=True,
        enable_shift_filter=True,
    )

    # Create filter instance
    face_filter = FaceFilter(config=filter_config)

    # Example face detection result
    face_result = {
        "bbox": [100, 100, 200, 200],
        "landmarks": [
            {"landmark": [120, 130]},  # left eye
            {"landmark": [180, 130]},  # right eye
            {"landmark": [150, 160]},  # nose
            {"landmark": [130, 190]},  # left mouth
            {"landmark": [170, 190]},  # right mouth
        ],
        "zone_info": [False, True, True, False],  # in zones 1,2 but not 0,3
        "track_id": 1,
    }

    # Apply filtering
    should_process = face_filter.should_process_face(face_result)
    print(f"Should process face: {should_process}")


# Example 2: Using FaceExtractGizmo with configuration object
def example_gizmo_with_config():
    """Example of using FaceExtractGizmo with filter configuration."""

    # Create custom filter configuration
    custom_filter_config = FaceFilterConfig(
        min_face_size=64,
        zone_ids=[2, 3],
        enable_frontal_filter=True,
        enable_shift_filter=False,  # Disable shift filtering
    )

    # Create gizmo with configuration
    gizmo = FaceExtractGizmo(
        target_image_size=112,
        filter_config=custom_filter_config,
        reid_config={"expiration_frames": 30},
    )

    print(f"Gizmo created with custom filter config: {custom_filter_config.to_dict()}")


# Example 3: Using FaceExtractGizmo with clean new interface
def example_gizmo_new_interface():
    """Example of using FaceExtractGizmo with new grouped interface."""

    # Create gizmo with new grouped interface
    gizmo = FaceExtractGizmo(
        target_image_size=112,
        filter_config=FaceFilterConfig(
            min_face_size=48,
            zone_ids=[1, 2],
            enable_frontal_filter=True,
            enable_shift_filter=False,
        ),
        reid_config={"expiration_frames": 45},
    )

    print("Gizmo created with new grouped interface")


# Example 4: Creating multiple filters for different scenarios
def example_multiple_filter_configs():
    """Example of creating different filter configurations for different scenarios."""

    # Strict filtering for high-quality face extraction
    strict_config = FaceFilterConfig(
        min_face_size=100,
        zone_ids=None,  # No zone filtering
        enable_frontal_filter=True,
        enable_shift_filter=True,
    )

    # Permissive filtering for face detection
    permissive_config = FaceFilterConfig(
        min_face_size=32,
        zone_ids=None,
        enable_frontal_filter=False,
        enable_shift_filter=False,
    )

    # Zone-specific filtering
    zone_specific_config = FaceFilterConfig(
        min_face_size=48,
        zone_ids=[1, 3, 5],  # Only specific zones
        enable_frontal_filter=True,
        enable_shift_filter=True,
    )

    # Create filters
    strict_filter = FaceFilter(config=strict_config)
    permissive_filter = FaceFilter(config=permissive_config)
    zone_filter = FaceFilter(config=zone_specific_config)

    print("Created multiple filter configurations:")
    print(f"  Strict: {strict_config.to_dict()}")
    print(f"  Permissive: {permissive_config.to_dict()}")
    print(f"  Zone-specific: {zone_specific_config.to_dict()}")


# Example 5: Dynamic filter configuration from external source
def example_dynamic_config():
    """Example of creating filter configuration from external data."""

    # Configuration from JSON/dict (e.g., from config file)
    config_data = {
        "min_face_size": 80,
        "zone_ids": [2, 4, 6],
        "enable_frontal_filter": True,
        "enable_shift_filter": False,
    }

    # Create configuration from dictionary
    dynamic_config = FaceFilterConfig.from_dict(config_data)

    # Create filter
    dynamic_filter = FaceFilter(config=dynamic_config)

    print(f"Dynamic configuration created: {dynamic_config.to_dict()}")


if __name__ == "__main__":
    print("=== FaceFilter Examples ===")
    print("\n1. Standalone Filter Usage:")
    example_standalone_filter()

    print("\n2. Gizmo with Configuration Object:")
    example_gizmo_with_config()

    print("\n3. Backward Compatible Usage:")
    example_gizmo_new_interface()

    print("\n4. Multiple Filter Configurations:")
    example_multiple_filter_configs()

    print("\n5. Dynamic Configuration:")
    example_dynamic_config()
