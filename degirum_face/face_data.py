#
# face_data.py: data structures for face tracking
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements data structures for managing face tracking state and object maps.
#

import threading
import copy
from typing import Dict, Any, List, Optional, ClassVar
from dataclasses import dataclass, asdict, field


@dataclass
class FaceAttributes:
    """
    Class to hold detected face attributes
    """

    attributes: Optional[Any]  # face attributes
    db_id: Optional[str] = None  # database ID
    embeddings: list = field(default_factory=list)  # list of embeddings for the face


@dataclass
class FaceRecognitionResult(FaceAttributes):
    """
    Class to hold face recognition results
    """

    bbox: Optional[Any] = None  # bounding box coordinates
    detection_score: Optional[float] = None  # face detection confidence score
    similarity_score: Optional[float] = None  # face similarity score
    landmarks: Optional[List[dict]] = None  # face landmarks

    @staticmethod
    def from_dict(result: Dict[str, Any]) -> "FaceRecognitionResult":
        """
        Create FaceRecognitionResult instance from object detection result dictionary augmented with face recognition data.

        Args:
            result: Dictionary containing face detection/recognition results as returned by face recognition pipeline

        Returns:
            FaceRecognitionResult: New instance initialized from result data
        """
        return FaceRecognitionResult(
            bbox=result.get("bbox"),
            detection_score=result.get("score"),
            similarity_score=result.get("face_similarity_score"),
            landmarks=result.get("landmarks"),
            attributes=result.get("face_attributes"),
            db_id=result.get("face_db_id"),
            embeddings=(
                [result.get("face_embeddings")]
                if result.get("face_embeddings") is not None
                else []
            ),
        )

    def __str__(self) -> str:
        """
        Pretty print the face recognition results.

        Returns:
            str: Formatted string representation of the results
        """
        lines = []

        # Attributes and ID
        lines.append(f"Attributes      : {self.attributes}")
        lines.append(f"Database ID     : {self.db_id}")

        # Scores
        if self.detection_score is not None:
            lines.append(f"Detection Score : {self.detection_score:.3f}")
        if self.similarity_score is not None:
            lines.append(f"Similarity Score: {self.similarity_score:.3f}")

        # Bounding box
        if self.bbox is not None:
            lines.append(
                f"Bounding box    : [{self.bbox[0]:.0f}, {self.bbox[1]:.0f}, {self.bbox[2]:.0f}, {self.bbox[3]:.0f}]"
            )

        # Landmarks
        if self.landmarks is not None:
            lines.append(f"Landmarks       : {len(self.landmarks)} points")

        # Embeddings
        if self.embeddings:
            lines.append(f"Embeddings      : {len(self.embeddings)} vector(s)")

        return "\n".join(lines)


@dataclass
class FaceStatus(FaceAttributes):
    """
    Class to hold detected face runtime status.
    """

    track_id: int = 0  # face track ID
    last_reid_frame: int = -1  # last frame number on which reID was performed
    next_reid_frame: int = -1  # next frame number on which reID should be performed
    confirmed_count: int = 0  # number of times the face was confirmed
    is_confirmed: bool = False  # whether the face status is confirmed
    is_alerted: bool = False  # whether the alert was triggered for this face

    # default labels
    lbl_not_tracked: ClassVar[str] = "not tracked"
    lbl_identifying: ClassVar[str] = "identifying"
    lbl_confirming: ClassVar[str] = "confirming"
    lbl_unknown: ClassVar[str] = "UNKNOWN"

    def __str__(self):
        return (
            str(self.attributes)
            if self.attributes is not None
            else FaceStatus.lbl_unknown
        )

    def to_dict(self):
        return asdict(self)


class ObjectMap:
    """Thread-safe map of object IDs to object attributes."""

    def __init__(self):
        """
        Constructor.
        """

        self._lock = threading.Lock()
        self.map: Dict[int, Any] = {}
        self.alert = False  # flag to indicate if an alert was triggered

    def set_alert(self, alert: bool = True) -> None:
        """
        Set the alert flag.

        Args:
            alert (bool): True to set the alert, False to reset it.
        """
        with self._lock:
            self.alert = alert

    def read_alert(self) -> bool:
        """
        Read the alert flag and reset it.

        Returns:
            bool: True if an alert was triggered, False otherwise.
        """
        with self._lock:
            alert = self.alert
            self.alert = False
            return alert

    def put(self, id: int, value: Any) -> None:
        """
        Add/update an object in the map

        Args:
            id (int): Object ID
            value (Any): Object attributes reference
        """
        with self._lock:
            self.map[id] = value

    def get(self, id: int) -> Optional[Any]:
        """
        Get the object by ID

        Args:
            id (int): The ID of the tracked face.

        Returns:
            Optional[Any]: The deep copy of object attributes or None if not found.
        """
        with self._lock:
            return copy.deepcopy(self.map.get(id))

    def delete(self, expr) -> Dict[int, Any]:
        """
        Delete objects from the map

        Args:
            expr (lambda): logical expression to filter objects to delete

        Returns:
            Dict[int, Any]: Map of deleted object IDs to their values.
        """
        with self._lock:
            deleted_items = {
                key: value for key, value in self.map.items() if expr(value)
            }
            for key in deleted_items.keys():
                del self.map[key]
            return deleted_items
