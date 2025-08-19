#
# face_utils.py: Face processing utility functions
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Standalone utility functions for face image processing operations.
#
# This module provides pure utility functions for face processing:
# - face_align_and_crop: Align faces to standard pose using ArcFace reference points
# - validate_landmarks: Validate landmark format and structure
# - landmarks_from_dict: Convert dictionary landmarks to numpy arrays
#
# These functions are stateless and can be used independently of the face tracking pipeline.
# They focus on computer vision operations rather than tracking coordination.
#
# Typical usage:
#     from degirum_face.face_utils import face_align_and_crop, landmarks_from_dict
#
#     # Convert landmarks from detection format
#     landmarks = landmarks_from_dict(detection_landmarks)
#
#     # Align and crop face for identification
#     aligned_face = face_align_and_crop(image, landmarks, 112)
#

import cv2
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .reid_database import ReID_Database

# Constants for face alignment
ARCFACE_REFERENCE_SIZE = 112  # Standard ArcFace reference image size
REQUIRED_LANDMARKS_COUNT = 5  # Number of landmarks required for alignment


# ArcFace reference keypoints for 112x112 image
# These coordinates define the standard face pose for alignment
# Order: left eye, right eye, nose, left mouth, right mouth
ARCFACE_REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],  # left eye center
        [73.5318, 51.5014],  # right eye center
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def face_align_and_crop(
    img: np.ndarray, landmarks: List[np.ndarray], image_size: int
) -> np.ndarray:
    """
    Align and crop a face from an image using facial landmarks.

    This function performs affine transformation to align the face to a standard
    reference pose and crops it to the specified size. Uses ArcFace reference
    keypoints for consistent alignment across different face images.

    The alignment process:
    1. Maps input landmarks to ArcFace reference points
    2. Estimates affine transformation matrix
    3. Applies transformation to align the face
    4. Crops to specified size with the face centered

    Args:
        img: The source image containing the face (full image, not pre-cropped)
        landmarks: List of 5 facial keypoints as (x, y) coordinates in order:
                  [left eye, right eye, nose, left mouth, right mouth]
        image_size: Target size for the output square image (width = height)

    Returns:
        Aligned and cropped face image of shape (image_size, image_size, 3)

    Raises:
        ValueError: If landmarks count != 5, image_size <= 0, or invalid input format
        RuntimeError: If affine transformation matrix estimation fails

    Example:
        >>> import numpy as np
        >>> from degirum_face.face_utils import face_align_and_crop
        >>>
        >>> # Facial landmarks from detection (5 points)
        >>> landmarks = [
        ...     np.array([120, 130]),  # left eye center
        ...     np.array([180, 130]),  # right eye center
        ...     np.array([150, 150]),  # nose tip
        ...     np.array([130, 170]),  # left mouth corner
        ...     np.array([170, 170]),  # right mouth corner
        ... ]
        >>>
        >>> # Align and crop to standard 112x112 size
        >>> aligned_face = face_align_and_crop(image, landmarks, 112)
        >>> print(aligned_face.shape)  # (112, 112, 3)
    """
    # Input validation
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(img)}")
    if img.size == 0:
        raise ValueError("Image is empty")

    if len(landmarks) != REQUIRED_LANDMARKS_COUNT:
        raise ValueError(
            f"Expected {REQUIRED_LANDMARKS_COUNT} landmarks, got {len(landmarks)}"
        )

    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    # Validate landmarks format
    if not validate_landmarks(landmarks):
        raise ValueError(
            "Invalid landmarks format. Expected list of 5 (x,y) points as numpy arrays"
        )

    # Scale ArcFace reference points to target image size
    dst = ARCFACE_REFERENCE_POINTS * (image_size / ARCFACE_REFERENCE_SIZE)

    # Convert landmarks to numpy array for OpenCV
    landmarks_array = np.array(landmarks, dtype=np.float32)

    # Estimate affine transformation matrix using LMEDS for robustness
    M, _ = cv2.estimateAffinePartial2D(
        landmarks_array, dst, method=cv2.LMEDS, confidence=0.99
    )

    if M is None:
        raise RuntimeError(
            "Failed to estimate affine transformation matrix. "
            "Check that landmarks are valid and not degenerate."
        )

    # Apply transformation and crop to final size
    aligned_img = cv2.warpAffine(
        img, M, (image_size, image_size), flags=cv2.INTER_LINEAR
    )
    return aligned_img


def validate_landmarks(landmarks: List[np.ndarray]) -> bool:
    """
    Validate facial landmarks format and structure.

    Checks that landmarks are properly formatted for face alignment:
    - Exactly 5 landmark points
    - Each landmark is a numpy array
    - Each landmark has exactly 2 coordinates (x, y)
    - Coordinates are finite numbers

    Args:
        landmarks: List of landmark points as numpy arrays

    Returns:
        True if all landmarks are valid, False otherwise

    Example:
        >>> import numpy as np
        >>> good_landmarks = [np.array([30, 40]), np.array([70, 40]),
        ...                   np.array([50, 60]), np.array([35, 80]), np.array([65, 80])]
        >>> validate_landmarks(good_landmarks)  # True
        >>>
        >>> bad_landmarks = [np.array([30]), np.array([70, 40])]  # Wrong count & shape
        >>> validate_landmarks(bad_landmarks)   # False
    """
    if not landmarks or len(landmarks) != REQUIRED_LANDMARKS_COUNT:
        return False

    try:
        for lm in landmarks:
            # Check type and shape
            if not isinstance(lm, np.ndarray) or lm.shape != (2,):
                return False

            # Check that coordinates are finite numbers
            if not np.all(np.isfinite(lm)):
                return False

    except (AttributeError, IndexError, TypeError):
        return False

    return True


def landmarks_from_dict(landmarks_dict: List[dict]) -> List[np.ndarray]:
    """
    Convert landmarks from dictionary format to numpy arrays.

    Transforms landmark data from the dictionary format typically returned by
    face detection models into the numpy array format required by alignment functions.

    Args:
        landmarks_dict: List of dictionaries, each with 'landmark' key containing [x, y] coordinates

    Returns:
        List of landmark points as numpy arrays with shape (2,) each

    Raises:
        ValueError: If input format is invalid or landmarks are missing
        KeyError: If 'landmark' key is missing from any dictionary

    Example:
        >>> # Typical format from face detection
        >>> detection_landmarks = [
        ...     {"landmark": [120.5, 130.2]},  # left eye
        ...     {"landmark": [180.1, 129.8]},  # right eye
        ...     {"landmark": [150.3, 155.0]},  # nose
        ...     {"landmark": [125.7, 175.4]},  # left mouth
        ...     {"landmark": [175.2, 174.9]},  # right mouth
        ... ]
        >>>
        >>> # Convert for use with face_align_and_crop
        >>> landmarks = landmarks_from_dict(detection_landmarks)
        >>> print(landmarks[0])  # array([120.5, 130.2])
    """
    if not landmarks_dict:
        raise ValueError("landmarks_dict cannot be empty")

    if len(landmarks_dict) != REQUIRED_LANDMARKS_COUNT:
        raise ValueError(
            f"Expected {REQUIRED_LANDMARKS_COUNT} landmark dictionaries, "
            f"got {len(landmarks_dict)}"
        )

    try:
        landmarks = []
        for i, lm_dict in enumerate(landmarks_dict):
            if not isinstance(lm_dict, dict):
                raise ValueError(
                    f"Landmark {i} must be a dictionary, got {type(lm_dict)}"
                )

            if "landmark" not in lm_dict:
                raise KeyError(f"Landmark {i} missing 'landmark' key")

            coords = lm_dict["landmark"]
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                raise ValueError(
                    f"Landmark {i} coordinates must be [x, y] list/tuple, got {coords}"
                )

            landmark_array = np.array(coords, dtype=np.float32)
            if not np.all(np.isfinite(landmark_array)):
                raise ValueError(f"Landmark {i} contains invalid coordinates: {coords}")

            landmarks.append(landmark_array)

        return landmarks

    except (TypeError, ValueError, KeyError) as e:
        raise ValueError(f"Invalid landmarks_dict format: {e}") from e


def get_landmark_names() -> List[str]:
    """
    Get the names of facial landmarks in order.

    Returns the standard names for the 5 facial landmarks used in face alignment,
    corresponding to the order expected by face_align_and_crop().

    Returns:
        List of landmark names in the expected order

    Example:
        >>> from degirum_face.face_utils import get_landmark_names
        >>> names = get_landmark_names()
        >>> print(names)
        ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    """
    return ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]


def landmarks_to_dict(landmarks: List[np.ndarray]) -> List[dict]:
    """
    Convert landmarks from numpy arrays back to dictionary format.

    Inverse operation of landmarks_from_dict(). Useful for serialization
    or when interfacing with systems that expect dictionary format.

    Args:
        landmarks: List of landmark points as numpy arrays

    Returns:
        List of dictionaries with 'landmark' key containing [x, y] coordinates

    Raises:
        ValueError: If landmarks are invalid format

    Example:
        >>> import numpy as np
        >>> landmarks = [np.array([120.5, 130.2]), np.array([180.1, 129.8])]
        >>> landmarks_dict = landmarks_to_dict(landmarks[:2])  # Convert first 2
        >>> print(landmarks_dict)
        [{'landmark': [120.5, 130.2]}, {'landmark': [180.1, 129.8]}]
    """
    if not validate_landmarks(landmarks):
        raise ValueError("Invalid landmarks format")

    return [{"landmark": lm.tolist()} for lm in landmarks]


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize face embedding to unit length.

    Extracted from FaceSearchGizmo.run() method (lines 688-689).
    Essential for consistent similarity calculations in face recognition.

    Args:
        embedding: Raw embedding vector from ReID model

    Returns:
        Normalized embedding with unit length

    Raises:
        ValueError: If embedding has zero norm (invalid embedding)

    Examples:
        >>> import numpy as np
        >>> raw_embedding = np.array([0.1, 0.2, 0.3])
        >>> normalized = normalize_embedding(raw_embedding)
        >>> print(f"Norm: {np.linalg.norm(normalized):.6f}")  # Should be 1.0
        Norm: 1.000000
    """
    # Flatten embedding and compute norm (extracted from FaceSearchGizmo)
    flattened = embedding.ravel()
    norm = np.linalg.norm(flattened)

    if norm == 0:
        raise ValueError("Embedding has zero norm - invalid embedding")

    return flattened / norm


def search_face_in_database(embedding: np.ndarray, database: "ReID_Database") -> tuple:
    """
    Search for face in database using normalized embedding.

    Extracted from FaceSearchGizmo.run() method (line 690).
    Combines embedding normalization with database lookup.

    Args:
        embedding: Raw embedding vector from ReID model
        database: ReID_Database instance for searching

    Returns:
        Tuple of (db_id, attributes) where:
        - db_id: Database identifier if found, None if not found
        - attributes: Person attributes if found, None if not found

    Examples:
        >>> from degirum_face.reid_database import ReID_Database
        >>> db = ReID_Database("faces.lance")
        >>> db_id, attrs = search_face_in_database(embedding_vector, db)
        >>> if db_id is not None:
        ...     print(f"Found person: {attrs.get('name', 'Unknown')}")
    """
    # Normalize embedding before search
    normalized_embedding = normalize_embedding(embedding)

    # Search database using normalized embedding
    return database.get_attributes_by_embedding(normalized_embedding)
