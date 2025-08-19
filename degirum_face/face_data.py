"""
Face Data Structures and Enumerations

This module defines core data structures and enumerations used throughout the
face tracking and recognition system. Provides standardized data containers
for face state management, tracking metadata, and alert configuration.

Core Components:
    - FaceStatus: Runtime status tracking for detected faces
    - AlertMode: Enumeration for alert trigger conditions

Data Management:
    The FaceStatus dataclass serves as the primary state container for each
    detected face throughout its lifecycle in the tracking system. It maintains:

    - Identity information (database ID, track ID)
    - ReID scheduling (frame counters for recognition attempts)
    - Confirmation tracking (detection reliability scoring)
    - Alert state (notification status)
    - Feature embeddings (face recognition vectors)

Integration Points:
    - ObjectMap: Uses FaceStatus as the tracked object type
    - TrackFilter: Manages next_reid_frame scheduling
    - FaceSearchGizmo: Updates database ID and confirmation status
    - FaceExtractGizmo: Creates initial FaceStatus instances
    - Alert System: Uses AlertMode for notification configuration

Thread Safety:
    FaceStatus instances are designed for use within the thread-safe ObjectMap
    container. Individual field updates should be atomic, but complex operations
    may require external synchronization.

Example Usage:
    >>> # Create new face status for tracking
    >>> face_status = FaceStatus(
    ...     attributes={"name": "John Doe", "department": "Engineering"},
    ...     track_id=12345,
    ...     confirmed_count=3,
    ...     is_confirmed=True
    ... )
    >>>
    >>> # Configure alert mode for unknown faces
    >>> alert_config = AlertMode.ON_UNKNOWNS
    >>>
    >>> # Serialize status for storage/transmission
    >>> status_dict = face_status.to_dict()

Copyright DeGirum Corporation 2025
"""

from typing import Any, Optional, ClassVar
from dataclasses import dataclass, asdict, field
from enum import Enum


@dataclass
class FaceStatus:
    """
    Runtime status container for tracked faces in the face recognition system.

    This dataclass maintains comprehensive state information for each detected face
    throughout its lifecycle in the tracking pipeline. Serves as the primary data
    structure for face identity management, ReID scheduling, and alert processing.

    State Categories:
        1. Identity: Database ID, track ID, and associated attributes
        2. ReID Scheduling: Frame-based timing for recognition attempts
        3. Confirmation: Reliability scoring and verification status
        4. Alerting: Notification state and trigger tracking
        5. Features: Face embedding vectors for recognition

    Lifecycle Management:
        - Created: When face is first detected (FaceExtractGizmo)
        - Updated: During ReID processing (FaceSearchGizmo)
        - Tracked: Throughout object lifetime (ObjectMap)
        - Serialized: For storage/transmission (to_dict method)

    Thread Safety:
        Individual field updates are atomic, but complex multi-field operations
        may require external synchronization when used in concurrent contexts.

    Attributes:
        attributes: User-defined metadata (name, department, etc.)
            - Typically populated from database lookup results
            - Can contain any JSON-serializable data
            - None indicates no associated metadata

        db_id: Unique identifier from face database
            - Set when face is recognized against database
            - None for unknown/unidentified faces
            - Used for database operations and lookups

        track_id: Unique identifier for tracking session
            - Assigned by tracking system for object continuity
            - Remains constant while face is visible
            - Different from database ID (identity vs. detection)

        last_reid_frame: Frame number of most recent ReID attempt
            - Updated each time face recognition is performed
            - Used for tracking ReID frequency and timing
            - -1 indicates no ReID attempts yet

        next_reid_frame: Scheduled frame for next ReID attempt
            - Calculated by TrackFilter based on exponential backoff
            - Prevents excessive ReID processing on stable tracks
            - -1 indicates ReID scheduling not yet determined

        confirmed_count: Number of consecutive confirmations
            - Incremented when face detection/recognition succeeds
            - Used for reliability scoring and filtering
            - Higher values indicate more stable detections

        is_confirmed: Whether face status is considered reliable
            - True when confirmed_count exceeds threshold
            - Affects alert generation and processing decisions
            - Reduces false positives from transient detections

        is_alerted: Whether alert has been triggered for this face
            - Prevents duplicate alert generation
            - Reset when face leaves tracking area
            - Used with AlertMode configuration

        embeddings: Collection of face feature vectors
            - Contains face recognition embedding arrays
            - Multiple embeddings improve recognition robustness
            - Used for similarity comparison and database matching

    Class Labels:
        Predefined string constants for common face status displays:
        - lbl_not_tracked: "not tracked" - Face not in tracking system
        - lbl_identifying: "identifying" - ReID in progress
        - lbl_confirming: "confirming" - Building confidence
        - lbl_unknown: "UNKNOWN" - No database match found

    Example:
        >>> # Create status for newly detected face
        >>> status = FaceStatus(
        ...     attributes=None,  # No metadata yet
        ...     track_id=42,
        ...     confirmed_count=1,
        ...     embeddings=[embedding_vector]
        ... )
        >>>
        >>> # Update after database lookup
        >>> status.db_id = "employee_123"
        >>> status.attributes = {"name": "Alice Smith", "role": "Manager"}
        >>> status.is_confirmed = True
        >>>
        >>> # Serialize for storage
        >>> data = status.to_dict()
        >>> print(f"Tracking: {status}")  # Uses __str__ method
    """

    attributes: Optional[Any]  # face attributes
    db_id: Optional[str] = None  # database ID
    track_id: int = 0  # face track ID
    last_reid_frame: int = -1  # last frame number on which reID was performed
    next_reid_frame: int = -1  # next frame number on which reID should be performed
    confirmed_count: int = 0  # number of times the face was confirmed
    is_confirmed: bool = False  # whether the face status is confirmed
    is_alerted: bool = False  # whether the alert was triggered for this face
    embeddings: list = field(default_factory=list)  # list of embeddings for the face

    # default labels
    lbl_not_tracked: ClassVar[str] = "not tracked"
    lbl_identifying: ClassVar[str] = "identifying"
    lbl_confirming: ClassVar[str] = "confirming"
    lbl_unknown: ClassVar[str] = "UNKNOWN"

    def __str__(self):
        """
        Return human-readable string representation of face status.

        Provides user-friendly display text based on the current face state.
        Prioritizes meaningful attributes over technical identifiers.

        Returns:
            Formatted string representation:
            - attributes value if available (e.g., name from database)
            - "UNKNOWN" label if no attributes are set

        Display Logic:
            - Database attributes (name, role, etc.) are most informative
            - Falls back to generic "UNKNOWN" for unidentified faces
            - Consistent with UI display requirements

        Example:
            >>> status1 = FaceStatus(attributes={"name": "John Doe"})
            >>> print(status1)  # Output: "{'name': 'John Doe'}"
            >>>
            >>> status2 = FaceStatus(attributes=None)
            >>> print(status2)  # Output: "UNKNOWN"
        """
        return (
            str(self.attributes)
            if self.attributes is not None
            else FaceStatus.lbl_unknown
        )

    def to_dict(self):
        """
        Convert FaceStatus instance to dictionary representation.

        Serializes all dataclass fields to a plain dictionary suitable for
        JSON encoding, database storage, or network transmission.

        Returns:
            Dictionary containing all field names and values:
            - Preserves data types (int, bool, list, etc.)
            - Handles None values appropriately
            - Maintains field ordering

        Use Cases:
            - JSON API responses
            - Database record storage
            - Inter-service communication
            - Configuration persistence
            - Debugging and logging

        Example:
            >>> status = FaceStatus(
            ...     db_id="emp_123",
            ...     track_id=42,
            ...     is_confirmed=True,
            ...     attributes={"name": "Alice"}
            ... )
            >>> data = status.to_dict()
            >>> # data = {
            >>> #     'attributes': {'name': 'Alice'},
            >>> #     'db_id': 'emp_123',
            >>> #     'track_id': 42,
            >>> #     'last_reid_frame': -1,
            >>> #     'next_reid_frame': -1,
            >>> #     'confirmed_count': 0,
            >>> #     'is_confirmed': True,
            >>> #     'is_alerted': False,
            >>> #     'embeddings': []
            >>> # }
        """
        return asdict(self)


class AlertMode(Enum):
    """
    Enumeration defining alert trigger conditions for face detection system.

    Controls when alert notifications are generated based on face recognition
    results. Provides flexible alert configuration for different security and
    monitoring scenarios.

    Alert Trigger Logic:
        The alert system evaluates each detected face against the configured
        AlertMode to determine if a notification should be sent. Alert generation
        considers both the face's database status and the current mode setting.

    Security Applications:
        - Access Control: Alert on unknown faces in secure areas
        - VIP Monitoring: Alert when known individuals are detected
        - Comprehensive Logging: Alert on all detections for audit trails
        - Silent Operation: Disable alerts for passive monitoring

    Integration:
        Used by FaceSearchGizmo and alert processing components to determine
        notification behavior. Combined with FaceStatus.is_alerted to prevent
        duplicate alerts for the same detection.

    Values:
        NONE (0): No alert generation
            - Silent monitoring mode
            - Face detection and recognition proceed normally
            - No notifications sent regardless of recognition results
            - Useful for passive observation or system testing

        ON_UNKNOWNS (1): Alert when unknown faces are detected
            - Triggers on faces not found in database
            - Security-focused mode for access control
            - Helps identify unauthorized individuals
            - Common for restricted area monitoring

        ON_KNOWNS (2): Alert when known faces are detected
            - Triggers on faces matched to database entries
            - VIP monitoring and special event tracking
            - Attendance verification systems
            - Executive or customer recognition scenarios

        ON_ALL (3): Alert on every face detection
            - Comprehensive monitoring mode
            - Generates notifications for all detected faces
            - Maximum visibility into face detection activity
            - Useful for high-security areas or detailed logging

    Performance Considerations:
        - Alert processing may impact system performance
        - Consider alert frequency and notification infrastructure capacity
        - ON_ALL mode generates highest alert volume
        - Combine with face filtering to reduce unnecessary alerts

    Example Usage:
        >>> # Configure for security monitoring
        >>> security_mode = AlertMode.ON_UNKNOWNS
        >>>
        >>> # VIP detection system
        >>> vip_mode = AlertMode.ON_KNOWNS
        >>>
        >>> # Comprehensive audit logging
        >>> audit_mode = AlertMode.ON_ALL
        >>>
        >>> # Silent operation
        >>> passive_mode = AlertMode.NONE
        >>>
        >>> # Check alert condition
        >>> face_is_known = face_status.db_id is not None
        >>> should_alert = (
        ...     (mode == AlertMode.ON_UNKNOWNS and not face_is_known) or
        ...     (mode == AlertMode.ON_KNOWNS and face_is_known) or
        ...     (mode == AlertMode.ON_ALL)
        ... )
    """

    NONE = 0  # no alert
    ON_UNKNOWNS = 1  # set alert on unknown faces
    ON_KNOWNS = 2  # set alert on known faces
    ON_ALL = 3  # set alert on all detected faces
