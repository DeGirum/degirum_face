#
# face_utils.py: utility functions for face processing
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements utility functions for face alignment, cropping, and landmark-based filtering.
#

import cv2
import numpy as np
from typing import List


def face_align_and_crop(
    img: np.ndarray, landmarks: List[np.ndarray], image_size: int
) -> np.ndarray:
    """
    Align and crop the face from the image based on the given landmarks.

    Args:
        img (np.ndarray): The full image (not the cropped bounding box).
        landmarks (List[np.ndarray]): List of 5 keypoints (landmarks) as (x, y) coordinates in the following order:
            [left eye, right eye, nose, left mouth, right mouth].
        image_size (int): The size to which the image should be resized.

    Returns:
        np.ndarray: the aligned face image
    """

    # reference keypoints for alignment:
    # these are the coordinates of the 5 keypoints in the reference image (112x112);
    # the order is: left eye, right eye, nose, left mouth, right mouth
    _arcface_ref_kps = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    assert len(landmarks) == 5
    dst = _arcface_ref_kps * image_size / 112.0  # scale to the target size

    M, _ = cv2.estimateAffinePartial2D(np.array(landmarks), dst, method=cv2.LMEDS)

    aligned_img = cv2.warpAffine(img, M, [image_size, image_size])
    return aligned_img


def face_is_frontal(landmarks: List[np.ndarray]) -> bool:
    """
    Check if the face is frontal based on the landmarks.

    Args:
        landmarks (List[np.ndarray]): List of 5 keypoints (landmarks) as (x, y) coordinates in the following order:
            [left eye, right eye, nose, left mouth, right mouth].

    Returns:
        bool: True if the face is frontal, False otherwise.
    """

    assert len(landmarks) == 5
    quad = np.array(
        [
            landmarks[0],  # left eye
            landmarks[1],  # right eye
            landmarks[4],  # right mouth
            landmarks[3],  # left mouth
        ],
        dtype=np.float32,
    )
    nose = landmarks[2]
    return cv2.pointPolygonTest(quad, tuple(nose), measureDist=False) > 0


def face_is_shifted(bbox: List[float], landmarks: List[np.ndarray]) -> bool:
    """
    Check if the face is shifted based on the landmarks.

    Args:
        bbox (List[float]): Bounding box of the face as [x1, y1, x2, y2].
        landmarks (List[np.ndarray]): List of keypoints (landmarks) as (x, y) coordinates

    Returns:
        bool: True if the face is shifted to a side of bbox, False otherwise.
    """

    assert len(bbox) == 4
    xc, yc = (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5

    return (
        all(x < xc for x, y in landmarks)
        or all(x >= xc for x, y in landmarks)
        or all(y < yc for x, y in landmarks)
        or all(y >= yc for x, y in landmarks)
    )
