"""
Unit tests demonstrating the improved testability of the refactored FaceFilter system.

These tests show how the standalone FaceFilter can be easily tested in isolation.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from degirum_face.face_tracking_gizmos import FaceFilter, FaceFilterConfig


class TestFaceFilter(unittest.TestCase):
    """Test cases for the standalone FaceFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_config = FaceFilterConfig()
        self.filter = FaceFilter(config=self.default_config)

        self.sample_face_result = {
            "bbox": [100, 100, 200, 200],
            "landmarks": [
                {"landmark": [120, 130]},  # left eye
                {"landmark": [180, 130]},  # right eye
                {"landmark": [150, 160]},  # nose
                {"landmark": [130, 190]},  # left mouth
                {"landmark": [170, 190]},  # right mouth
            ],
        }

    def test_filter_config_creation(self):
        """Test filter configuration creation and conversion."""
        config = FaceFilterConfig(
            min_face_size=50,
            zone_ids=[1, 2, 3],
            enable_frontal_filter=True,
            enable_shift_filter=False,
        )

        # Test to_dict conversion
        config_dict = config.to_dict()
        expected_dict = {
            "min_face_size": 50,
            "zone_ids": [1, 2, 3],
            "enable_frontal_filter": True,
            "enable_shift_filter": False,
        }
        self.assertEqual(config_dict, expected_dict)

        # Test from_dict creation
        new_config = FaceFilterConfig.from_dict(config_dict)
        self.assertEqual(new_config.min_face_size, 50)
        self.assertEqual(new_config.zone_ids, [1, 2, 3])

    def test_filter_with_missing_landmarks(self):
        """Test filter behavior with missing or invalid landmarks."""
        # No landmarks
        face_result = {"bbox": [100, 100, 200, 200]}
        self.assertFalse(self.filter.should_process_face(face_result))

        # Wrong number of landmarks
        face_result["landmarks"] = [{"landmark": [120, 130]}]  # Only 1 landmark
        self.assertFalse(self.filter.should_process_face(face_result))

    def test_size_filter(self):
        """Test size filtering functionality."""
        # Create filter with minimum size requirement
        config = FaceFilterConfig(min_face_size=50)
        size_filter = FaceFilter(config=config)

        # Test face that passes size filter
        large_face = self.sample_face_result.copy()
        large_face["bbox"] = [100, 100, 200, 200]  # 100x100 face
        self.assertTrue(size_filter._passes_size_filter(large_face))

        # Test face that fails size filter
        small_face = self.sample_face_result.copy()
        small_face["bbox"] = [100, 100, 130, 130]  # 30x30 face
        self.assertFalse(size_filter._passes_size_filter(small_face))

        # Test with no bbox
        no_bbox_face = self.sample_face_result.copy()
        del no_bbox_face["bbox"]
        self.assertTrue(
            size_filter._passes_size_filter(no_bbox_face)
        )  # Should pass if no bbox

    def test_zone_filter(self):
        """Test zone filtering functionality."""
        # Create filter with zone requirements
        config = FaceFilterConfig(zone_ids=[1, 2])
        zone_filter = FaceFilter(config=config)

        # Mock the ZoneCounter key
        with patch("degirum_face.face_tracking_gizmos.degirum_tools") as mock_tools:
            mock_tools.ZoneCounter.key_in_zone = "in_zone"

            # Test face in correct zone
            face_in_zone = self.sample_face_result.copy()
            face_in_zone["in_zone"] = [False, True, False, False]  # In zone 1
            self.assertTrue(zone_filter._passes_zone_filter(face_in_zone))

            # Test face not in any specified zone
            face_not_in_zone = self.sample_face_result.copy()
            face_not_in_zone["in_zone"] = [True, False, False, True]  # In zones 0,3
            self.assertFalse(zone_filter._passes_zone_filter(face_not_in_zone))

    def test_frontal_filter_integration(self):
        """Test frontal filter integration with static methods."""
        config = FaceFilterConfig(enable_frontal_filter=True)
        frontal_filter = FaceFilter(config=config)

        # Mock the static method to return True (frontal)
        with patch(
            "degirum_face.face_tracking_gizmos.FaceExtractGizmo.face_is_frontal",
            return_value=True,
        ):
            face_result = self.sample_face_result.copy()
            # Should pass frontal filter
            keypoints = [
                np.array([120, 130]),
                np.array([180, 130]),
                np.array([150, 160]),
                np.array([130, 190]),
                np.array([170, 190]),
            ]
            self.assertTrue(frontal_filter._is_face_frontal(keypoints))

    def test_shift_filter_integration(self):
        """Test shift filter integration with static methods."""
        config = FaceFilterConfig(enable_shift_filter=True)
        shift_filter = FaceFilter(config=config)

        # Mock the static method to return False (not shifted)
        with patch(
            "degirum_face.face_tracking_gizmos.FaceExtractGizmo.face_is_shifted",
            return_value=False,
        ):
            bbox = [100, 100, 200, 200]
            keypoints = [
                np.array([120, 130]),
                np.array([180, 130]),
                np.array([150, 160]),
                np.array([130, 190]),
                np.array([170, 190]),
            ]
            self.assertFalse(shift_filter._is_face_shifted(bbox, keypoints))

    def test_complete_filter_chain(self):
        """Test the complete filter chain with all filters enabled."""
        config = FaceFilterConfig(
            min_face_size=50,
            zone_ids=[1, 2],
            enable_frontal_filter=True,
            enable_shift_filter=True,
        )
        complete_filter = FaceFilter(config=config)

        # Create a face that should pass all filters
        good_face = {
            "bbox": [100, 100, 200, 200],  # Large enough
            "landmarks": [
                {"landmark": [120, 130]},
                {"landmark": [180, 130]},
                {"landmark": [150, 160]},
                {"landmark": [130, 190]},
                {"landmark": [170, 190]},
            ],
        }

        # Mock external dependencies
        with patch(
            "degirum_face.face_tracking_gizmos.degirum_tools"
        ) as mock_tools, patch(
            "degirum_face.face_tracking_gizmos.FaceExtractGizmo.face_is_frontal",
            return_value=True,
        ), patch(
            "degirum_face.face_tracking_gizmos.FaceExtractGizmo.face_is_shifted",
            return_value=False,
        ):

            mock_tools.ZoneCounter.key_in_zone = "in_zone"
            good_face["in_zone"] = [False, True, False, False]  # In zone 1

            # Should pass all filters
            self.assertTrue(complete_filter.should_process_face(good_face))

    def test_filter_with_kwargs_backward_compatibility(self):
        """Test that the filter can be created with kwargs for backward compatibility."""
        # Create filter using kwargs instead of config object
        filter_kwargs = FaceFilter(
            min_face_size=64,
            zone_ids=[2, 3],
            enable_frontal_filter=False,
            enable_shift_filter=True,
        )

        # Verify the config was created correctly
        self.assertEqual(filter_kwargs.config.min_face_size, 64)
        self.assertEqual(filter_kwargs.config.zone_ids, [2, 3])
        self.assertFalse(filter_kwargs.config.enable_frontal_filter)
        self.assertTrue(filter_kwargs.config.enable_shift_filter)


class TestFaceFilterErrorHandling(unittest.TestCase):
    """Test error handling in face filtering."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = FaceFilter()

    def test_frontal_filter_error_handling(self):
        """Test that frontal filter handles errors gracefully."""
        # Test with invalid landmarks that would cause an exception
        invalid_keypoints = [np.array([0, 0])]  # Wrong number of landmarks

        # Should return False when an error occurs
        self.assertFalse(self.filter._is_face_frontal(invalid_keypoints))

    def test_shift_filter_error_handling(self):
        """Test that shift filter handles errors gracefully."""
        # Test with invalid bbox that would cause an exception
        invalid_bbox = [100, 100]  # Wrong number of coordinates
        valid_keypoints = [np.array([120, 130]) for _ in range(5)]

        # Should return True (consider shifted) when an error occurs
        self.assertTrue(self.filter._is_face_shifted(invalid_bbox, valid_keypoints))


if __name__ == "__main__":
    unittest.main()
