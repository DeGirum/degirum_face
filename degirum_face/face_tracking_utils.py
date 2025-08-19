#
# face_tracking_utils.py: utility classes for face tracking
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Face tracking utility classes for managing face identification state and scheduling.
#
# This module provides:
# - ObjectMap: Thread-safe storage for object tracking results (generic, reusable)
# - TrackFilter: Smart scheduler to avoid redundant face re-identification processing
#
# Typical usage for face tracking:
#     face_map = ObjectMap[FaceStatus]()  # Type-safe storage for face data
#     if face_map is not None:  # Only create when coordination needed
#         track_filter = TrackFilter(face_map, expiration_frames=120)
#         if track_filter.should_reid_and_register(track_id, frame_id):
#             # Process face for identification
#             pass
#

import threading
import copy
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, TYPE_CHECKING

from .face_data import FaceStatus

if TYPE_CHECKING:
    from .face_data import AlertMode

T = TypeVar("T")  # Generic type parameter for ObjectMap


class ObjectMap(Generic[T]):
    """
    Thread-safe storage for object tracking results and metadata.

    This class stores object data by ID and provides thread-safe access for concurrent
    processing components. It's used to coordinate between different processing stages
    and share object identification/tracking results.

    Key features:
    - Thread-safe operations for concurrent access
    - Deep copying to prevent data races between components
    - Alert system for notification when specific objects are detected
    - Bulk deletion with custom filter expressions
    - Type safety through Generic[T] - ensures all stored objects are of type T

    Generic design: Specify the object type when creating the map for type safety.
    Example: ObjectMap[FaceStatus]() for face tracking, ObjectMap[CarStatus]() for car tracking.
    """

    def __init__(self):
        """
        Initialize an empty ObjectMap with thread synchronization.

        Creates:
        - Empty dictionary for storing object data by ID (typed as T)
        - Threading lock for safe concurrent access
        - Alert flag for detection event notifications
        """
        self._lock = threading.Lock()
        self.map: Dict[int, T] = {}  # Type-safe storage for objects of type T
        self.alert = False  # Global alert flag for detection events

    def set_alert(self, alert: bool = True) -> None:
        """
        Set the global alert flag for detection events.

        Used by processing components to signal when specific objects are detected
        based on application-specific criteria.

        Args:
            alert (bool): True to trigger alert, False to clear it.
        """
        with self._lock:
            self.alert = alert

    def read_alert(self) -> bool:
        """
        Read and reset the alert flag in one atomic operation.

        This is used by notification systems to check if any detection
        events occurred since the last check.

        Returns:
            bool: True if an alert was triggered since last read, False otherwise.
        """
        with self._lock:
            alert = self.alert
            self.alert = False
            return alert

    def put(self, id: int, value: T) -> None:
        """
        Store or update object data for a given ID.

        Args:
            id (int): Object ID (application-specific identifier)
            value (T): Object data of the specified generic type T
        """
        with self._lock:
            self.map[id] = value

    def get(self, id: int) -> Optional[T]:
        """
        Retrieve object data for a given ID.

        Returns a deep copy to prevent data races when multiple components
        access the same object data concurrently. This is safer but slower
        than shallow copying.

        Args:
            id (int): Object ID to look up

        Returns:
            Optional[T]: Deep copy of object data of type T or None if not found
        """
        with self._lock:
            return copy.deepcopy(self.map.get(id))

    def delete(self, expr: Callable[[T], bool]) -> None:
        """
        Remove objects from storage based on a filter condition.

        Commonly used to remove expired objects or objects that meet certain criteria.

        Args:
            expr (Callable[[T], bool]): Function that takes object data of type T
                                       and returns True to delete.
                                       Example: lambda obj: obj.timestamp + max_age < current_time
        """
        with self._lock:
            keys_to_delete = [key for key, value in self.map.items() if expr(value)]
            for key in keys_to_delete:
                del self.map[key]


class TrackFilter:
    """
    Smart scheduler for face re-identification to optimize processing performance.

    Problem: Re-identifying faces every frame is computationally expensive and unnecessary.
    Most faces remain stable for multiple frames, so constant re-processing wastes resources.

    Solution: Use exponential backoff scheduling:
    - Process new faces immediately (frame 0)
    - For known faces, gradually increase intervals: 1, 2, 4, 8, 16... frames
    - Cap maximum interval to ensure faces don't become stale
    - Remove faces that haven't been seen for too long

    Face-specific ObjectMap usage:
    - Key: track_id (int) - Face tracking ID from detection system
    - Value: FaceStatus object containing:
        * attributes: Identification result (name, etc.) or None for unknown
        * track_id: Face tracking ID for coordination
        * last_reid_frame: Last frame when face was processed for identification
        * next_reid_frame: Next frame when face should be processed again
        * confirmed_count: Number of times face was consistently identified
        * is_confirmed: Whether identification is confident enough
        * embeddings: List of face feature vectors (if accumulating)

    Frame-based timing:
    - frame_id: Monotonically increasing frame counter from video stream
    - Scheduling decisions based on frame numbers, not wall-clock time
    - Allows consistent behavior regardless of video frame rate

    Benefits:
    - Catches new faces quickly for immediate identification
    - Reduces CPU load by avoiding redundant processing of stable faces
    - Adapts processing frequency based on face stability
    - Maintains accuracy while improving performance

    Example timeline for track_id=123:
        Frame 100: ✅ Process (new face)     → next check at frame 101
        Frame 101: ✅ Process               → next check at frame 103 (+2)
        Frame 102: ❌ Skip (too soon)
        Frame 103: ✅ Process               → next check at frame 107 (+4)
        Frame 104-106: ❌ Skip
        Frame 107: ✅ Process               → next check at frame 115 (+8)
        ...and so on with exponential backoff
    """

    def __init__(
        self, face_reid_map: ObjectMap[FaceStatus], reid_expiration_frames: int
    ):
        """
        Initialize track scheduler with guaranteed coordination.

        Args:
            face_reid_map: Shared storage for face data. Must contain FaceStatus objects.
            reid_expiration_frames: Remove faces not seen for this many frames.
                                   0 means never expire (keep forever).
        """
        self._map = face_reid_map
        self._max = reid_expiration_frames

    def should_reid_and_register(self, track_id: int, frame_id: int) -> bool:
        """
        Atomically determine if a face should be processed and register it if new.

        This is the main entry point for track scheduling decisions. It combines
        the timing check and new face registration to avoid race conditions.

        Decision logic:
        1. New face? → Register and process immediately
        2. Existing face? → Check if enough time has passed based on exponential backoff

        Args:
            track_id: Face tracking ID from detection system
            frame_id: Current video frame number

        Returns:
            bool: True if face should be processed for identification, False to skip
        """
        face: Optional[FaceStatus] = self._map.get(track_id)
        if face is None:
            # New face - create tracking entry and process immediately
            face = FaceStatus(
                attributes=None,  # No identification yet
                track_id=track_id,
                last_reid_frame=frame_id,  # Remember when we processed it
                next_reid_frame=frame_id
                + 1,  # Next check in 1 frame (immediate follow-up)
            )
            self._map.put(track_id, face)
            return True

        # Existing face - check if enough time has passed
        if frame_id < face.next_reid_frame:
            return False  # Too soon, skip this frame

        # Time for reprocessing - update schedule with exponential backoff
        prev_interval = max(1, face.next_reid_frame - face.last_reid_frame)
        delta = 2 * prev_interval  # Double the previous interval (exponential backoff)
        if self._max > 0:
            delta = min(self._max, delta)  # Cap at maximum to prevent infinite delays

        face.last_reid_frame = frame_id
        face.next_reid_frame = frame_id + delta
        self._map.put(track_id, face)
        return True

    def expire(self, frame_id: int) -> None:
        """
        Remove faces that haven't been seen for too long.

        Cleanup operation to prevent memory leaks and remove stale face data.
        Faces are considered expired if they haven't been processed for
        reid_expiration_frames number of frames.

        Face expiration logic:
        - Compares face.last_reid_frame + max_age < current_frame_id
        - Uses ObjectMap.delete() with lambda to filter expired faces
        - Only runs when max_age > 0 (expiration enabled)

        Args:
            frame_id: Current video frame number for expiration calculation
        """
        if self._max > 0:
            # Remove faces where: last_seen + max_age < current_frame
            self._map.delete(lambda face: face.last_reid_frame + self._max < frame_id)


# Face recognition processing functions extracted from FaceSearchGizmo


def process_basic_recognition(face_obj: dict, db_id, attributes) -> None:
    """
    Process face recognition in recognition-only mode.

    Extracted from FaceSearchGizmo._process_recognition_only() method.
    Provides immediate recognition results without state management.
    Suitable for basic face recognition pipelines without tracking.

    Args:
        face_obj: Face object to update with recognition results
        db_id: Database ID from recognition (None if not found)
        attributes: Person attributes from database (None if not found)

    Note:
        In recognition-only mode, this function:
        - Adds recognition results directly to face object
        - Always sets is_confirmed=True (immediate confirmation)
        - Does not use credence counting
        - Does not store persistent state
        - Does not generate alerts
        - Does not accumulate embeddings

    Examples:
        >>> face_obj = {"track_id": 123, "bbox": [100, 100, 200, 200]}
        >>> process_basic_recognition(face_obj, "person_456", {"name": "John"})
        >>> print(face_obj["db_id"])  # "person_456"
        >>> print(face_obj["is_confirmed"])  # True
    """
    # Add recognition results directly to face object
    face_obj["db_id"] = db_id
    face_obj["attributes"] = attributes
    face_obj["is_confirmed"] = True  # Always confirmed in recognition-only mode


def process_tracked_recognition(
    track_id: Optional[int],
    db_id,
    attributes,
    embedding,
    face_reid_map: ObjectMap[FaceStatus],
    credence_count: int,
    accumulate_embeddings: bool,
    alert_mode: "AlertMode",
    alert_once: bool,
) -> None:
    """
    Process face recognition with full state management and credence counting.

    Extracted from FaceSearchGizmo._process_with_state_management() method.
    Uses ObjectMap for persistent tracking and implements all advanced features.

    Args:
        track_id: Tracking ID (None if tracking not available)
        db_id: Database ID from recognition
        attributes: Person attributes from database
        embedding: Normalized face embedding
        face_reid_map: Shared storage for face tracking state
        credence_count: Number of confirmations required before considering face identified
        accumulate_embeddings: Whether to store embeddings in face object
        alert_mode: Type of faces to generate alerts for
        alert_once: Whether to generate alert only once per face

    Features:
        - Credence counting for reliable identification
        - Persistent state management via ObjectMap
        - Alert generation based on recognition results
        - Embedding accumulation for quality improvement
        - Coordination between tracking and recognition components

    Examples:
        >>> from degirum_face.face_data import AlertMode
        >>> process_tracked_recognition(
        ...     track_id=123,
        ...     db_id="person_456",
        ...     attributes={"name": "John"},
        ...     embedding=normalized_embedding,
        ...     face_reid_map=face_map,
        ...     credence_count=3,
        ...     accumulate_embeddings=True,
        ...     alert_mode=AlertMode.ON_UNKNOWNS,
        ...     alert_once=True
        ... )
    """
    if track_id is not None and face_reid_map is not None:
        # We have tracking - use ReID coordination
        face = face_reid_map.get(track_id)
        if face is not None:
            # existing face - update the attributes
            if face.db_id == db_id:
                face.confirmed_count += 1
            else:
                face.confirmed_count = 1
                # reset frame counter when the face changes status for quick reconfirming
                face.next_reid_frame = face.last_reid_frame + 1

            face.is_confirmed = face.confirmed_count >= credence_count
            if face.attributes != attributes and not alert_once:
                face.is_alerted = False  # reset alert if attributes changed
            face.attributes = attributes
            face.db_id = db_id
            if accumulate_embeddings:
                face.embeddings.append(embedding)

            if face.is_confirmed:
                # Import here to avoid circular dependency
                from .face_data import AlertMode

                if (
                    (
                        alert_mode == AlertMode.ON_UNKNOWNS
                        and attributes is None
                        and not face.is_alerted
                    )
                    or (
                        alert_mode == AlertMode.ON_KNOWNS
                        and attributes is not None
                        and not face.is_alerted
                    )
                    or (alert_mode == AlertMode.ON_ALL and not face.is_alerted)
                ):
                    face_reid_map.set_alert(True)
                    face.is_alerted = True

            face_reid_map.put(track_id, face)
