#
# test_face_utils.py: Unit tests for face utility functions
# Copyright DeGirum Corp. 2025
#
# Unit tests for face_align_and_crop, face_is_frontal, and face_is_shifted functions
# (AI-generated)
#

import pytest
import numpy as np
import cv2
from degirum_face.face_utils import (
    face_align_and_crop,
    face_is_frontal,
    face_is_shifted,
)


def test_face_utils():
    """Test face utility functions"""

    # Test 1: face_align_and_crop function

    # Create a test image (100x100 RGB image)
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Define test landmarks (5 keypoints: left eye, right eye, nose, left mouth, right mouth)
    landmarks = [
        [30, 35],  # left eye
        [60, 35],  # right eye
        [45, 50],  # nose
        [35, 70],  # left mouth
        [55, 70],  # right mouth
    ]

    # Test with different image sizes
    for image_size in [64, 112, 224]:
        aligned_face = face_align_and_crop(test_img, landmarks, image_size)

        # Verify output dimensions
        assert aligned_face.shape == (image_size, image_size, 3)
        assert aligned_face.dtype == test_img.dtype

        # Verify the aligned image is not empty/black
        assert np.sum(aligned_face) > 0

    # Test assertion error for wrong number of landmarks
    with pytest.raises(AssertionError):
        face_align_and_crop(test_img, landmarks[:4], 112)  # Only 4 landmarks

    with pytest.raises(AssertionError):
        face_align_and_crop(
            test_img, landmarks + [np.array([10, 10])], 112
        )  # 6 landmarks

    # Test 2: face_is_frontal function

    # Test case: frontal face (nose inside the eye-mouth quadrilateral)
    frontal_landmarks = [
        [30, 40],  # left eye
        [70, 40],  # right eye
        [50, 55],  # nose (centered)
        [35, 75],  # left mouth
        [65, 75],  # right mouth
    ]

    assert face_is_frontal(frontal_landmarks) == True

    # Test case: profile/side face (nose outside the eye-mouth quadrilateral)
    profile_landmarks = [
        [30, 40],  # left eye
        [70, 40],  # right eye
        [20, 55],  # nose (far left, outside quad)
        [35, 75],  # left mouth
        [65, 75],  # right mouth
    ]

    assert face_is_frontal(profile_landmarks) == False

    # Test case: another profile orientation
    profile_landmarks2 = [
        [30, 40],  # left eye
        [70, 40],  # right eye
        [85, 55],  # nose (far right, outside quad)
        [35, 75],  # left mouth
        [65, 75],  # right mouth
    ]

    assert face_is_frontal(profile_landmarks2) == False

    # Test assertion error for wrong number of landmarks
    with pytest.raises(AssertionError):
        face_is_frontal(frontal_landmarks[:4])

    # Test 3: face_is_shifted function

    # Define a test bounding box [x1, y1, x2, y2]
    bbox = [20, 30, 80, 90]  # center at (50, 60)

    # Test case: normally distributed landmarks (not shifted)
    normal_landmarks = [
        [35, 45],  # left side
        [65, 45],  # right side
        [50, 55],  # center
        [40, 75],  # left-bottom
        [60, 75],  # right-bottom
    ]

    assert face_is_shifted(bbox, normal_landmarks) == False

    # Test case: all landmarks shifted to the left
    left_shifted_landmarks = [
        [25, 45],  # all x < center_x (50)
        [35, 45],
        [30, 55],
        [25, 75],
        [40, 75],
    ]

    assert face_is_shifted(bbox, left_shifted_landmarks) == True

    # Test case: all landmarks shifted to the right
    right_shifted_landmarks = [
        [55, 45],  # all x >= center_x (50)
        [65, 45],
        [60, 55],
        [55, 75],
        [70, 75],
    ]

    assert face_is_shifted(bbox, right_shifted_landmarks) == True

    # Test case: all landmarks shifted upward
    up_shifted_landmarks = [
        [35, 35],  # all y < center_y (60)
        [65, 40],
        [50, 45],
        [40, 50],
        [60, 55],
    ]

    assert face_is_shifted(bbox, up_shifted_landmarks) == True

    # Test case: all landmarks shifted downward
    down_shifted_landmarks = [
        [35, 65],  # all y >= center_y (60)
        [65, 70],
        [50, 75],
        [40, 80],
        [60, 85],
    ]

    assert face_is_shifted(bbox, down_shifted_landmarks) == True

    # Test assertion error for wrong bbox format
    with pytest.raises(AssertionError):
        face_is_shifted([20, 30, 80], normal_landmarks)  # Only 3 coordinates

    # Test 4: Edge cases and robustness

    # Test with minimal image size for face_align_and_crop
    minimal_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    minimal_landmarks = [
        [2, 3],
        [7, 3],
        [4, 5],
        [3, 7],
        [6, 7],
    ]

    aligned_minimal = face_align_and_crop(minimal_img, minimal_landmarks, 32)
    assert aligned_minimal.shape == (32, 32, 3)

    # Test face_is_frontal with landmarks forming a very small quadrilateral
    tiny_landmarks = [
        [50, 50],  # left eye
        [51, 50],  # right eye (very close)
        [50.5, 50.5],  # nose (in center)
        [50, 51],  # left mouth
        [51, 51],  # right mouth
    ]

    # Should handle small quadrilaterals gracefully
    result = face_is_frontal(tiny_landmarks)
    assert isinstance(result, (bool, np.bool_))

    # Test face_is_shifted with landmarks exactly on the center line
    center_bbox = [0, 0, 100, 100]  # center at (50, 50)
    edge_case_landmarks = [
        [50, 25],  # exactly on center x
        [50, 30],
        [50, 40],
        [50, 60],
        [50, 75],
    ]

    # All landmarks have x >= center_x (50), so should be considered shifted
    assert face_is_shifted(center_bbox, edge_case_landmarks) == True

    # Test with floating point coordinates
    float_landmarks = [
        [30.5, 40.7],
        [69.3, 39.8],
        [49.9, 54.2],
        [35.1, 74.6],
        [64.8, 75.1],
    ]

    float_bbox = [20.0, 30.0, 80.0, 90.0]

    # Should handle floating point coordinates
    assert isinstance(face_is_frontal(float_landmarks), (bool, np.bool_))
    assert isinstance(face_is_shifted(float_bbox, float_landmarks), (bool, np.bool_))

    # Test grayscale image for face_align_and_crop
    gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Should work with 2D grayscale images too
    aligned_gray = face_align_and_crop(gray_img, landmarks, 64)
    assert aligned_gray.shape == (64, 64)
