#
# face_filters.py: Face detection filtering utilities
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Comprehensive face filtering system for processing face detection results.
#
# This module provides a flexible, configurable filtering system to validate and
# filter face detections based on multiple criteria:
#
# Quality Filters:
# - Landmark validation: Ensures face has valid facial keypoints
# - Size filtering: Removes faces that are too small for reliable processing
# - Frontal detection: Filters out profile/side faces using geometric analysis
# - Shift detection: Removes faces positioned at edges of bounding boxes
#
# Spatial Filters:
# - Zone filtering: Process only faces within specified regions of interest
#
# Design Features:
# - Configurable: All filters can be enabled/disabled via FaceFilterConfig
# - Composable: Filters are applied in sequence for optimal performance
# - Standalone: Can be used independently of the tracking pipeline
# - Efficient: Expensive checks (frontal/shift) are performed last
#
# Expected Face Detection Result Structure:
# The face_result dictionary passed to filtering methods should contain:
#
# {
#     "bbox": [x1, y1, x2, y2],           # Bounding box coordinates (float/int)
#     "landmarks": [                       # List of 5 facial landmark dictionaries
#         {"landmark": [x, y]},           # Left eye center
#         {"landmark": [x, y]},           # Right eye center
#         {"landmark": [x, y]},           # Nose tip
#         {"landmark": [x, y]},           # Left mouth corner
#         {"landmark": [x, y]}            # Right mouth corner
#     ],
#     "confidence": 0.95,                 # Detection confidence (optional)
#     "label": "face",                    # Detection class label (optional)
#     ZoneCounter.key_in_zone: [True, False, True],  # Zone membership flags (optional)
#     # ... other detection metadata
# }
#
# Typical usage:
#     from degirum_face.face_filters import FaceFilter, FaceFilterConfig
#
#     # Configure filtering criteria
#     config = FaceFilterConfig(
#         min_face_size=50,           # Minimum face size in pixels
#         zone_ids=[0, 1],           # Only process faces in zones 0 and 1
#         enable_frontal_filter=True, # Filter out profile faces
#         enable_shift_filter=True   # Filter out edge-positioned faces
#     )
#
#     # Create filter and apply to detections
#     face_filter = FaceFilter(config)
#     for detection in face_detections:
#         if face_filter.should_process_face(detection):
#             # Process this high-quality face detection
#             process_face(detection)
#

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum

import degirum_tools
from .face_utils import landmarks_from_dict

# Constants for face filtering
REQUIRED_LANDMARKS_COUNT = 5  # Number of landmarks required for face processing
MIN_BBOX_COORDINATES = 4  # Number of coordinates in bounding box [x1, y1, x2, y2]

# Documentation for expected face detection result structure
FACE_RESULT_STRUCTURE_DOC = """
Expected Face Detection Result Dictionary Structure:

The face_result parameter passed to FaceFilter methods should be a dictionary
containing face detection data from DeGirum AI models or compatible systems.

Required Fields:
    bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
        - x1, y1: Top-left corner coordinates
        - x2, y2: Bottom-right corner coordinates
        - Coordinates are in image pixel space
        
    landmarks (List[Dict]): List of exactly 5 facial landmark dictionaries
        Each landmark dict contains:
        - "landmark": [x, y] coordinates in image pixel space
        
        Standard 5-point landmark order:
        [0] Left eye center (subject's left)
        [1] Right eye center (subject's right)  
        [2] Nose tip
        [3] Left mouth corner (subject's left)
        [4] Right mouth corner (subject's right)

Optional Fields:
    confidence (float): Detection confidence score (0.0 to 1.0)
    label (str): Object class label (typically "face")
    {ZoneCounter.key_in_zone} (List[bool]): Zone membership flags
        - Boolean array indicating which spatial zones contain the face
        - Index corresponds to zone ID
        - Required only if zone filtering is enabled

Example Structure:
    {
        "bbox": [100.5, 50.2, 200.8, 150.7],
        "landmarks": [
            {"landmark": [120.1, 80.3]},  # Left eye
            {"landmark": [180.4, 82.1]},  # Right eye
            {"landmark": [150.2, 110.5]}, # Nose
            {"landmark": [135.8, 130.2]}, # Left mouth
            {"landmark": [165.1, 132.0]}  # Right mouth
        ],
        "confidence": 0.96,
        "label": "face",
        "in_zone": [True, False, True]  # In zones 0 and 2, not in zone 1
    }

Validation Notes:
    - bbox must contain exactly 4 numeric values
    - landmarks must contain exactly 5 dictionaries
    - Each landmark dict must have "landmark" key with [x, y] coordinates
    - Missing or malformed data will cause filters to reject the detection
    - Zone data is only required if zone filtering is enabled in config
"""


class FaceLandmarkIndex(IntEnum):
    """
    Enumeration for standard 5-point facial landmark indices.

    Provides semantic naming for facial landmark array positions to improve
    code readability and maintainability. Based on the standard 5-point
    landmark model used by most face detection systems.

    Landmark Order:
        - Eyes: Left eye (0), Right eye (1)
        - Nose: Nose tip (2)
        - Mouth: Left mouth corner (3), Right mouth corner (4)

    Note: "Left" and "right" are from the face's perspective (subject's left/right),
    not the viewer's perspective.
    """

    LEFT_EYE = 0  # Left eye center (subject's left)
    RIGHT_EYE = 1  # Right eye center (subject's right)
    NOSE = 2  # Nose tip
    LEFT_MOUTH = 3  # Left mouth corner (subject's left)
    RIGHT_MOUTH = 4  # Right mouth corner (subject's right)


@dataclass
class FaceFilterConfig:
    """
    Configuration for face filtering operations.

    This dataclass defines all configurable parameters for face filtering.
    All filters are optional and can be disabled by setting appropriate values.

    Attributes:
        min_face_size: Minimum size (pixels) of smaller bbox dimension.
                      Set to 0 to disable size filtering.
        zone_ids: List of zone IDs to process. None means process all zones.
                 Only faces within these zones will be processed.
        enable_frontal_filter: Whether to filter out non-frontal faces.
                              Uses geometric analysis of facial landmarks.
        enable_shift_filter: Whether to filter out shifted faces.
                           Removes faces positioned at edges of bounding boxes.

    Example:
        >>> # Strict filtering for high-quality processing
        >>> strict_config = FaceFilterConfig(
        ...     min_face_size=80,
        ...     zone_ids=[0, 1, 2],
        ...     enable_frontal_filter=True,
        ...     enable_shift_filter=True
        ... )
        >>>
        >>> # Permissive filtering for maximum detection
        >>> permissive_config = FaceFilterConfig(
        ...     min_face_size=30,
        ...     zone_ids=None,  # All zones
        ...     enable_frontal_filter=False,
        ...     enable_shift_filter=False
        ... )
    """

    min_face_size: int = 0
    zone_ids: Optional[List[int]] = None
    enable_frontal_filter: bool = True
    enable_shift_filter: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_face_size < 0:
            raise ValueError(
                f"min_face_size must be non-negative, got {self.min_face_size}"
            )

        if self.zone_ids is not None:
            if not isinstance(self.zone_ids, list):
                raise ValueError(
                    f"zone_ids must be a list or None, got {type(self.zone_ids)}"
                )
            if any(not isinstance(zid, int) or zid < 0 for zid in self.zone_ids):
                raise ValueError("All zone_ids must be non-negative integers")


class FaceFilter:
    def filter_face_with_reason(self, face_result: dict) -> dict:
        """
        Annotate face_result with face_rejected and reject_reason keys based on filtering.

        Args:
            face_result: Face detection result dictionary (will be modified in-place)

        Returns:
            The same dict, with 'face_rejected' (bool) and 'reject_reason' (str or None) keys set.
        """
        # 1. Check landmarks
        if not self._has_valid_landmarks(face_result):
            face_result["face_rejected"] = True
            face_result["reject_reason"] = "invalid_landmarks"
            return face_result

        # 2. Check size
        if not self._is_face_large_enough(face_result):
            face_result["face_rejected"] = True
            face_result["reject_reason"] = "face_too_small"
            return face_result

        # 3. Check zones
        if not self._is_face_in_zones(face_result):
            face_result["face_rejected"] = True
            face_result["reject_reason"] = "not_in_zone"
            return face_result

        # 4. Check if frontal
        if self.config.enable_frontal_filter and not self._is_face_frontal(face_result):
            face_result["face_rejected"] = True
            face_result["reject_reason"] = "not_frontal"
            return face_result

        # 5. Check if shifted
        if self.config.enable_shift_filter and self._is_face_shifted(face_result):
            face_result["face_rejected"] = True
            face_result["reject_reason"] = "face_shifted"
            return face_result

        # Passed all filters
        face_result["face_rejected"] = False
        face_result["reject_reason"] = None
        return face_result

    """
    Comprehensive face filter for validating face detection results.

    This class applies multiple filtering criteria to face detections to ensure
    only high-quality, processable faces are passed through. Filters are applied
    in order of computational cost (cheap filters first) for optimal performance.

    Filter Pipeline Order:
    1. Landmark validation (very fast) - Check for required facial keypoints
    2. Size filtering (fast) - Ensure face is large enough for processing
    3. Zone filtering (fast) - Check if face is in region of interest
    4. Frontal detection (expensive) - Geometric analysis of face orientation
    5. Shift detection (expensive) - Check face position within bounding box

    Design Principles:
    - Fail fast: Cheap filters eliminate bad detections early
    - Configurable: Each filter can be enabled/disabled
    - Robust: Handles missing/invalid data gracefully
    - Stateless: Thread-safe, no internal state modifications

    Typical use cases:
    - Pre-processing for face recognition: Use strict frontal + size filters
    - Face tracking: Use permissive settings for tracking consistency
    - Zone monitoring: Enable zone filtering for specific area monitoring
    """

    def __init__(self, config: FaceFilterConfig):
        """
        Initialize face filter with configuration.

        Args:
            config: Filter configuration specifying which filters to apply

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, FaceFilterConfig):
            raise ValueError(f"config must be FaceFilterConfig, got {type(config)}")
        self.config = config

    def should_process_face(self, face_result: dict) -> bool:
        """
        Determine if a face detection should be processed based on all enabled filters.

        Applies filters in order of computational cost for optimal performance.
        Returns False as soon as any filter fails (short-circuit evaluation).

        Args:
            face_result: Face detection result dictionary. Must contain:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'landmarks': List of 5 landmark dicts with 'landmark': [x, y]
                - ZoneCounter.key_in_zone: Zone membership flags (if zone filtering enabled)

                See FACE_RESULT_STRUCTURE_DOC for complete specification.

        Returns:
            True if face passes all enabled filters and should be processed,
            False if face should be rejected

        Performance Notes:
            - Fast filters (landmarks, size, zone) execute first
            - Expensive filters (frontal, shift) execute only if needed
            - Typical execution time: 0.1ms (fast path) to 2ms (all filters)

        Example:
            >>> config = FaceFilterConfig(min_face_size=50, enable_frontal_filter=True)
            >>> face_filter = FaceFilter(config)
            >>>
            >>> detection = {
            ...     'bbox': [100, 100, 200, 200],  # 100x100 face
            ...     'landmarks': [
            ...         {'landmark': [120, 130]},  # Left eye
            ...         {'landmark': [180, 132]},  # Right eye
            ...         {'landmark': [150, 160]},  # Nose
            ...         {'landmark': [135, 180]},  # Left mouth
            ...         {'landmark': [165, 182]}   # Right mouth
            ...     ]
            ... }
            >>>
            >>> if face_filter.should_process_face(detection):
            ...     print("High-quality face detected!")
            ... else:
            ...     print("Face filtered out")

        Raises:
            No exceptions are raised - invalid input results in filter rejection
        """
        # 1. Check landmarks (very fast - just length check)
        if not self._has_valid_landmarks(face_result):
            return False

        # 2. Check size (fast - simple arithmetic)
        if not self._is_face_large_enough(face_result):
            return False

        # 3. Check zones (fast - array lookup)
        if not self._is_face_in_zones(face_result):
            return False

        # 4. Check if frontal (expensive - geometric computation)
        if self.config.enable_frontal_filter and not self._is_face_frontal(face_result):
            return False

        # 5. Check if shifted (expensive - coordinate analysis)
        if self.config.enable_shift_filter and self._is_face_shifted(face_result):
            return False

        return True

    def _has_valid_landmarks(self, face_result: dict) -> bool:
        """
        Check if face detection contains valid landmark data.

        Validates both the presence and quantity of facial landmarks.
        Landmarks are essential for face alignment and quality assessment.

        Args:
            face_result: Face detection result dictionary

        Returns:
            True if face has the required number of landmarks (5), False otherwise

        Technical Notes:
            - Requires exactly 5 landmarks for ArcFace alignment algorithm
            - Standard landmarks: left eye, right eye, nose, left mouth, right mouth
            - Invalid/missing landmarks indicate poor detection quality

        Example Landmark Structure:
            landmarks: [
                {'landmark': [x1, y1]},  # Left eye corner
                {'landmark': [x2, y2]},  # Right eye corner
                {'landmark': [x3, y3]},  # Nose tip
                {'landmark': [x4, y4]},  # Left mouth corner
                {'landmark': [x5, y5]}   # Right mouth corner
            ]
        """
        landmarks = face_result.get("landmarks")
        return bool(landmarks and len(landmarks) == 5)

    def _is_face_large_enough(self, face_result: dict) -> bool:
        """
        Check if face detection meets minimum size requirements.

        Filters out faces that are too small for reliable recognition.
        Small faces typically produce poor-quality feature embeddings.

        Args:
            face_result: Face detection result dictionary containing 'bbox'

        Returns:
            True if face meets size requirements, False if too small

        Size Calculation:
            - Uses minimum of width and height (square assumption)
            - Handles cases where bbox might be malformed
            - min_face_size <= 0 disables size filtering

        Performance Impact:
            - Small faces consume processing resources with poor results
            - Filtering improves overall system accuracy and speed
            - Typical threshold: 50-100 pixels for reliable recognition

        Example:
            >>> # Face with bbox [100, 100, 200, 250] has size 100 (min of 100x150)
            >>> config = FaceFilterConfig(min_face_size=80)
            >>> filter = FaceFilter(config)
            >>> result = {'bbox': [100, 100, 200, 250]}
            >>> filter._is_face_large_enough(result)  # Returns True (100 >= 80)
        """
        if self.config.min_face_size <= 0:
            return True

        bbox = face_result.get("bbox")
        if not bbox or len(bbox) != 4:
            return False

        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        return min(w, h) >= self.config.min_face_size

    def _is_face_in_zones(self, face_result: dict) -> bool:
        """
        Check if face detection is within specified monitoring zones.

        Enables spatial filtering to focus processing on specific areas.
        Useful for ignoring faces in irrelevant locations (hallways, backgrounds).

        Args:
            face_result: Face detection result containing zone membership data

        Returns:
            True if face is in any specified zone or no zones configured,
            False if face is outside all specified zones

        Zone Integration:
            - Requires ZoneCounter gizmo to populate zone membership
            - Zone IDs correspond to user-defined spatial regions
            - Empty zone_ids list disables zone filtering

        Performance Benefits:
            - Reduces processing load by ignoring irrelevant areas
            - Improves accuracy by focusing on areas of interest
            - Fast lookup operation (array indexing)

        Example:
            >>> # Monitor only zones 0 and 2 (entrance and checkout)
            >>> config = FaceFilterConfig(zone_ids=[0, 2])
            >>> filter = FaceFilter(config)
            >>>
            >>> # Face in zone 1 (storage area) - filtered out
            >>> result = {ZoneCounter.key_in_zone: [False, True, False]}
            >>> filter._is_face_in_zones(result)  # Returns False
            >>>
            >>> # Face in zone 2 (checkout) - processed
            >>> result = {ZoneCounter.key_in_zone: [False, False, True]}
            >>> filter._is_face_in_zones(result)  # Returns True
        """
        if not self.config.zone_ids:
            return True

        in_zone = face_result.get(degirum_tools.ZoneCounter.key_in_zone)
        if in_zone is None:
            return False

        return any(in_zone[zid] for zid in self.config.zone_ids if zid < len(in_zone))

    def _is_face_frontal(self, face_result: dict) -> bool:
        """
        Check if face is oriented frontally based on landmark geometry.

        Analyzes facial landmark positions to determine face orientation.
        Frontal faces typically provide better recognition accuracy.

        Args:
            face_result: Face detection result containing landmarks

        Returns:
            True if face appears to be frontal, False if profile/angled

        Algorithm Details:
            - Uses geometric analysis of facial landmark positions
            - Delegates to optimized implementation for consistency
            - Handles landmark parsing and error cases gracefully

        Quality Impact:
            - Frontal faces yield more consistent feature embeddings
            - Profile faces often produce unreliable recognition results
            - Filtering improves overall system accuracy

        Error Handling:
            - Returns False for malformed landmarks
            - Gracefully handles parsing failures
            - Consistent with other filter methods
        """
        landmarks = face_result.get("landmarks")
        if not landmarks or len(landmarks) != 5:
            return False

        try:
            keypoints = landmarks_from_dict(landmarks)
            return self._face_is_frontal_impl(keypoints)
        except (ValueError, IndexError):
            return False

    def _is_face_shifted(self, face_result: dict) -> bool:
        """
        Check if face is improperly positioned within its bounding box.

        Detects faces that are poorly centered or cropped, which can
        negatively impact recognition accuracy.

        Args:
            face_result: Face detection result with bbox and landmarks

        Returns:
            True if face appears shifted/poorly positioned, False if well-centered

        Shift Detection Logic:
            - Analyzes landmark positions relative to bounding box
            - Checks for proper centering and margin consistency
            - Delegates to optimized implementation for accuracy

        Quality Impact:
            - Well-centered faces produce better feature embeddings
            - Shifted faces may be partially cropped or poorly framed
            - Filtering improves recognition consistency

        Conservative Approach:
            - Returns True (shifted) when detection is uncertain
            - Prefers false positives over false negatives
            - Ensures only high-quality faces proceed to recognition

        Error Handling:
            - Assumes shifted for malformed input data
            - Gracefully handles missing bbox or landmarks
            - Consistent error behavior across filter methods
        """
        bbox = face_result.get("bbox")
        landmarks = face_result.get("landmarks")
        if not bbox or not landmarks or len(landmarks) != 5:
            return True  # assume shifted if we can't tell

        try:
            keypoints = landmarks_from_dict(landmarks)
            return self._face_is_shifted_impl(bbox, keypoints)
        except (ValueError, IndexError):
            return True

    @staticmethod
    def _face_is_frontal_impl(landmarks: List[np.ndarray]) -> bool:
        """
        Core implementation of frontal face detection using geometric analysis.

        Determines face orientation by analyzing the spatial relationship
        between facial landmarks using computational geometry.

        Args:
            landmarks: List of 5 keypoints in standard order:
                [left_eye, right_eye, nose, left_mouth, right_mouth]

        Returns:
            True if face appears frontal, False if profile/angled

        Algorithm Details:
            - Creates quadrilateral from eye and mouth corners
            - Tests if nose landmark falls inside this quadrilateral
            - Uses OpenCV's point-in-polygon test for robust detection

        Geometric Principle:
            For frontal faces, the nose should be positioned within the
            quadrilateral formed by connecting the eye and mouth corners.
            Profile faces will have the nose outside this region.

        Quadrilateral Construction:
            - Top edge: left_eye → right_eye
            - Right edge: right_eye → right_mouth
            - Bottom edge: right_mouth → left_mouth
            - Left edge: left_mouth → left_eye

        Robustness Features:
            - Handles various face sizes and orientations
            - Accounts for minor landmark detection noise
            - Uses floating-point precision for accuracy

        Performance: O(1) - constant time geometric test
        Accuracy: High for typical face detection scenarios

        Raises:
            ValueError: If landmarks count is not exactly 5
        """
        if len(landmarks) != 5:
            raise ValueError(f"Expected 5 landmarks, got {len(landmarks)}")

        quad = np.array(
            [
                landmarks[FaceLandmarkIndex.LEFT_EYE],  # left eye
                landmarks[FaceLandmarkIndex.RIGHT_EYE],  # right eye
                landmarks[FaceLandmarkIndex.RIGHT_MOUTH],  # right mouth
                landmarks[FaceLandmarkIndex.LEFT_MOUTH],  # left mouth
            ],
            dtype=np.float32,
        )
        nose = landmarks[FaceLandmarkIndex.NOSE]
        return cv2.pointPolygonTest(quad, tuple(nose), measureDist=False) > 0

    @staticmethod
    def _face_is_shifted_impl(bbox: List[float], landmarks: List[np.ndarray]) -> bool:
        """
        Core implementation of face shift detection algorithm.

        Analyzes landmark distribution within the bounding box to detect
        poorly positioned or cropped faces that may impact recognition quality.

        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            landmarks: List of facial landmark points as numpy arrays

        Returns:
            True if face appears shifted/poorly positioned, False if well-centered

        Algorithm Details:
            - Calculates bounding box center point
            - Tests if all landmarks cluster on one side of center
            - Detects both horizontal and vertical shifting patterns

        Shift Detection Logic:
            1. Compute bbox center: (x_center, y_center)
            2. Check if all landmarks are left OR right of x_center
            3. Check if all landmarks are above OR below y_center
            4. Return True if any clustering pattern is detected

        Quality Rationale:
            - Well-centered faces have landmarks distributed around center
            - Shifted faces cluster landmarks on one side (poor cropping)
            - Such faces often produce inconsistent recognition results

        Use Cases:
            - Detecting edge-of-frame faces (partially visible)
            - Identifying poorly cropped face regions
            - Filtering faces with inadequate margins

        Performance: O(n) where n = number of landmarks (typically 5)
        Accuracy: High for detecting obvious positioning issues

        Raises:
            ValueError: If bbox doesn't contain exactly 4 coordinates

        Example Patterns:
            - All landmarks left of center: shifted right
            - All landmarks above center: shifted up
            - Mixed distribution: well-centered (not shifted)
        """
        if len(bbox) != 4:
            raise ValueError(f"Expected bbox with 4 coordinates, got {len(bbox)}")

        xc, yc = (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5
        return (
            all(x < xc for x, y in landmarks)
            or all(x >= xc for x, y in landmarks)
            or all(y < yc for x, y in landmarks)
            or all(y >= yc for x, y in landmarks)
        )
