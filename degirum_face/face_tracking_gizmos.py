#
# face_tracking_gizmos.py: face tracking gizmo classes implementation
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements gizmo classes for face extraction, alignment, and re-identification (reID).
#

import cv2
import numpy as np
import copy
from typing import List, Dict, Any, Optional, Union

import degirum_tools
from degirum_tools.streams import (
    Gizmo,
    StreamData,
    VideoSourceGizmo,
    tag_inference,
    tag_video,
)

from .reid_database import ReID_Database
from .face_data import FaceStatus, AlertMode
from .face_tracking_utils import (
    ObjectMap,
    TrackFilter,
    process_basic_recognition,
    process_tracked_recognition,
)
from .face_filters import FaceFilter, FaceFilterConfig
from .face_utils import (
    face_align_and_crop,
    landmarks_from_dict,
    search_face_in_database,
)


tag_obj_annotate = "object_annotate"  # tag for object annotation meta
tag_face_align = "face_align"  # tag for face alignment and cropping meta
tag_face_search = "face_search"  # tag for face search meta


class ObjectAnnotateGizmo(Gizmo):
    """Object annotating gizmo"""

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_obj_annotate, tag_inference]

    def __init__(
        self,
        object_map: ObjectMap[FaceStatus],
        *,
        label_map: dict = {},
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Constructor.

        Args:
            object_map (ObjectMap[FaceStatus]): Shared storage containing face identification results.
                This is the same ObjectMap used by other gizmos in the pipeline.
                The gizmo reads identification results to display appropriate labels.
            label_map (dict): Map of special labels (FaceStatus.lbl_*) to their display names.
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._object_map = object_map
        self._label_map = label_map
        self._label_map.setdefault(
            FaceStatus.lbl_not_tracked, FaceStatus.lbl_not_tracked
        )
        self._label_map.setdefault(
            FaceStatus.lbl_identifying, FaceStatus.lbl_identifying
        )
        self._label_map.setdefault(FaceStatus.lbl_confirming, FaceStatus.lbl_confirming)
        self._label_map.setdefault(FaceStatus.lbl_unknown, FaceStatus.lbl_unknown)

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have face detection gizmo in upstream"
                )

            clone = degirum_tools.clone_result(result)

            for r in clone.results:
                track_id = r.get("track_id")
                if track_id is None:
                    r["label"] = self._label_map[FaceStatus.lbl_not_tracked]
                else:
                    obj_status = self._object_map.get(track_id)
                    if obj_status is None:
                        r["label"] = self._label_map[FaceStatus.lbl_identifying]
                    else:
                        if obj_status.is_confirmed:
                            if obj_status.attributes is None:
                                # unknown face
                                r["label"] = self._label_map[FaceStatus.lbl_unknown]
                            else:
                                # known face
                                r["attributes"] = obj_status.attributes
                                r["label"] = str(obj_status)
                        else:
                            # face is not confirmed yet
                            r["label"] = self._label_map[FaceStatus.lbl_confirming]

            new_meta = data.meta.clone()
            new_meta.append(clone, self.get_tags())
            self.send_result(StreamData(data, new_meta))


class FaceExtractGizmo(Gizmo):
    """Face extracting and aligning gizmo"""

    # meta keys
    key_original_result = "original_result"  # original AI object detection result
    key_cropped_result = "cropped_result"  # sub-result for particular crop
    key_cropped_index = "cropped_index"  # the number of that sub-result
    key_is_last_crop = "is_last_crop"  # 'last crop in the frame' flag

    def __init__(
        self,
        target_image_size: int,
        *,
        # Face filtering configuration
        frame_filter: Optional[FaceFilterConfig] = None,
        # Processing mode
        tracking_enabled: bool = True,
        # ReID coordination configuration
        track_filter: Optional[Dict[str, Any]] = None,
        # Stream configuration
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Initialize face extraction and alignment gizmo with grouped configuration.

        Args:
            target_image_size: Size (width/height) for aligned face crops in pixels.
                Typical values: 112 (ArcFace), 224 (general recognition)

            frame_filter: Face filtering configuration (FaceFilterConfig).
                Controls which faces are processed based on quality criteria.
                If None, no filtering is applied (all faces processed for maximum performance).

            tracking_enabled: If True, enables face tracking with ReID coordination.
                Optimized for tracking pipelines with state management.
                When False, processes all faces without tracking (basic recognition mode).

            track_filter: ReID coordination configuration dictionary containing:
                - 'object_map': ObjectMap for face state coordination (Optional[ObjectMap[FaceStatus]])
                - 'expiration_frames': Frames before re-processing face (int, default: 0)

                Example:
                    {
                        'object_map': shared_face_map,
                        'expiration_frames': 30  # Re-process every 30 frames
                    }

                If None, no coordination is applied (all faces processed for maximum performance).
                Ignored when tracking_enabled=False.

            stream_depth: Maximum frames buffered in processing pipeline.

            allow_drop: Whether to drop frames under high load conditions.

        Raises:
            ValueError: If target_image_size <= 0 or track_filter is malformed

        Examples:
            >>> # Basic usage with default filtering
            >>> gizmo = FaceExtractGizmo(target_image_size=112)
            >>>
            >>> # With custom filtering
            >>> gizmo = FaceExtractGizmo(
            ...     target_image_size=224,
            ...     frame_filter=FaceFilterConfig(
            ...         min_face_size=80,
            ...         zone_ids=[0, 1, 2],
            ...         enable_frontal_filter=True,
            ...         enable_shift_filter=False
            ...     )
            ... )
            >>>
            >>> # With ReID coordination
            >>> gizmo = FaceExtractGizmo(
            ...     target_image_size=112,
            ...     tracking_enabled=True,
            ...     track_filter={
            ...         'object_map': face_map,
            ...         'expiration_frames': 30
            ...     }
            ... )
            >>>
            >>> # Complete configuration
            >>> gizmo = FaceExtractGizmo(
            ...     target_image_size=112,
            ...     frame_filter=FaceFilterConfig(min_face_size=50, zone_ids=[1, 2]),
            ...     tracking_enabled=True,
            ...     track_filter={'object_map': face_map, 'expiration_frames': 60},
            ...     stream_depth=20,
            ...     allow_drop=True
            ... )
        """
        super().__init__([(stream_depth, allow_drop)])

        # Validate target image size
        if target_image_size <= 0:
            raise ValueError("target_image_size must be positive")

        self._image_size = target_image_size
        self._tracking_enabled = tracking_enabled

        # Parse ReID configuration (ignored when tracking_enabled=False)
        face_reid_map = None
        reid_expiration_frames = 0

        if tracking_enabled and track_filter is not None:
            if not isinstance(track_filter, dict):
                raise ValueError("track_filter must be a dictionary")

            face_reid_map = track_filter.get("object_map")
            reid_expiration_frames = track_filter.get("expiration_frames", 0)

            if reid_expiration_frames < 0:
                raise ValueError("track_filter.expiration_frames must be non-negative")

        # Warn if track_filter provided when tracking disabled
        if not tracking_enabled and track_filter is not None:
            import warnings

            warnings.warn(
                "track_filter provided but tracking_enabled=False, ReID coordination will be bypassed"
            )

        # Configure face filtering - make it truly optional
        if frame_filter is not None:
            self._face_filter = FaceFilter(frame_filter)
        else:
            self._face_filter = None  # No filtering - process all faces

        # Configure ReID coordination - disabled when tracking disabled
        if tracking_enabled and face_reid_map is not None:
            self._track_filter = TrackFilter(face_reid_map, reid_expiration_frames)
        else:
            self._track_filter = None  # No coordination - process all faces

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_align]

    def run(self):
        """Run gizmo - clean and simple flow."""
        for data in self.get_input(0):
            if self._abort:
                break

            try:
                # Get input data
                result, frame_id = self._get_input_data(data)

                # Process each face
                for i, face_result in enumerate(result.results):
                    # Apply face filters if configured
                    if (
                        self._face_filter is not None
                        and not self._face_filter.should_process_face(face_result)
                    ):
                        continue

                    # Handle tracking requirements
                    if self._tracking_enabled:
                        track_id = face_result.get("track_id")
                        if track_id is None:
                            continue  # Skip faces without track_id when tracking is enabled

                        # Check ReID coordination requirements
                        if (
                            self._track_filter is not None
                            and not self._should_process_track(track_id, frame_id)
                        ):
                            continue

                    # Process this face
                    crop_img = self._extract_face_crop(data.data, face_result)
                    self._send_face_result(
                        crop_img, data, result, face_result, i, len(result.results)
                    )

                # Clean up expired faces
                self._cleanup_expired_faces(frame_id)

            except Exception as e:
                # Re-raise with context for better debugging
                raise RuntimeError(
                    f"{self.__class__.__name__} processing failed at frame {frame_id}"
                ) from e

    def _get_input_data(self, data: StreamData) -> tuple:
        """Get inference result and frame ID needed for face processing."""
        # get inference result
        result = data.meta.find_last(tag_inference)
        if result is None:
            raise Exception(
                f"{self.__class__.__name__}: inference meta not found: you need to have face detection gizmo in upstream"
            )

        # get current frame ID
        video_meta = data.meta.find_last(tag_video)
        if video_meta is None:
            raise Exception(
                f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream"
            )
        frame_id = video_meta[VideoSourceGizmo.key_frame_id]

        return result, frame_id

    def _should_process_track(self, track_id: int, frame_id: int) -> bool:
        """
        Determine if a track should be processed based on ReID filtering logic.

        Args:
            track_id: The face track ID (guaranteed to be valid when this method is called)
            frame_id: Current frame number

        Returns:
            bool: True if track should be processed, False otherwise
        """
        # If no ReID coordination, process all tracks (maximum performance)
        if self._track_filter is None:
            return True

        # Atomically check and register to avoid race conditions
        return self._track_filter.should_reid_and_register(track_id, frame_id)

    def _extract_face_crop(self, image: np.ndarray, face_result: dict) -> np.ndarray:
        """Extract and align a single face from the image."""
        landmarks = face_result.get("landmarks")
        keypoints = landmarks_from_dict(landmarks)
        return face_align_and_crop(image, keypoints, self._image_size)

    def _send_face_result(
        self,
        crop_img: np.ndarray,
        data: StreamData,
        result: Any,
        face_result: dict,
        face_index: int,
        total_faces: int,
    ) -> None:
        """Send face crop result through gizmo stream."""
        # Create crop metadata
        crop_obj = copy.deepcopy(face_result)
        crop_meta = self._create_crop_metadata(
            result, crop_obj, face_index, total_faces
        )

        # Send result
        new_meta = data.meta.clone()
        new_meta.remove_last(tag_inference)
        new_meta.append(crop_meta, self.get_tags())
        self.send_result(StreamData(crop_img, new_meta))

    def _create_crop_metadata(
        self, result: Any, crop_obj: dict, face_index: int, total_faces: int
    ) -> dict:
        """Create metadata for the cropped face."""
        return {
            self.key_original_result: result,
            self.key_cropped_result: crop_obj,
            self.key_cropped_index: face_index,
            self.key_is_last_crop: face_index == total_faces - 1,
        }

    def _cleanup_expired_faces(self, frame_id: int) -> None:
        """Delete expired faces from the ReID map."""
        if self._track_filter is not None:
            self._track_filter.expire(frame_id)


class FaceSearchGizmo(Gizmo):
    """Face reID search gizmo"""

    def __init__(
        self,
        face_reid_map: Optional[ObjectMap[FaceStatus]],
        db: ReID_Database,
        *,
        credence_count: int,
        stream_depth: int = 10,
        allow_drop: bool = False,
        accumulate_embeddings: bool = False,
        alert_mode: AlertMode = AlertMode.ON_UNKNOWNS,
        alert_once: bool = True,
        tracking_enabled: bool = True,
    ):
        """
        Constructor.

        Args:
            face_reid_map (Optional[ObjectMap[FaceStatus]]): Shared storage for face identification results.
                This is the same ObjectMap used by FaceExtractGizmo for coordination.
                The gizmo stores identification results here so other components can access them.
                Can be None when tracking_enabled=False for basic face recognition without tracking.
            db (ReID_Database): vector database object
            credence_count (int): Number of times the face is recognized before confirming it.
                Ignored when tracking_enabled=False.
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
            accumulate_embeddings (bool): Whether to accumulate embeddings in the face map.
                Ignored when tracking_enabled=False.
            alert_mode (AlertMode): Mode of alerting for the face search.
            alert_once (bool): Whether to trigger the alert only once for the given face.
                Ignored when tracking_enabled=False.
            tracking_enabled (bool): If True, enables face tracking with state management.
                Suitable for tracking pipelines with credence counting and persistent state.
                If False, performs immediate recognition without state management.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._face_reid_map = face_reid_map
        self._db = db
        self._credence_count = credence_count
        self._accumulate_embeddings = accumulate_embeddings
        self._alert_mode = alert_mode
        self._alert_once = alert_once
        self._tracking_enabled = tracking_enabled

        # Validate configuration
        if tracking_enabled and face_reid_map is None:
            raise ValueError("face_reid_map is required when tracking_enabled=True")
        if not tracking_enabled and face_reid_map is not None:
            import warnings

            warnings.warn(
                "face_reid_map provided but tracking_enabled=False, state management will be bypassed"
            )

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_search]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            # get inference result
            result = data.meta.find_last(tag_inference)
            if (
                result is None
                or not result.results
                or result.results[0].get("data") is None
            ):
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have reID inference gizmo in upstream"
                )

            # get current frame ID
            video_meta = data.meta.find_last(tag_video)
            if video_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream"
                )

            # get face crop result
            crop_meta = data.meta.find_last(tag_face_align)
            if crop_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: crop meta not found: you need to have {FaceExtractGizmo.__class__.__name__} in upstream"
                )

            face_obj = crop_meta.get(FaceExtractGizmo.key_cropped_result)
            assert face_obj

            track_id = face_obj.get("track_id")

            # In tracking mode, track_id may or may not be present

            # search the database for the face embedding (using extracted utility)
            embedding = result.results[0].get("data")
            db_id, attributes = search_face_in_database(embedding, self._db)

            # Handle recognition based on mode
            if not self._tracking_enabled:
                # Basic recognition mode: immediate results without state management
                process_basic_recognition(face_obj, db_id, attributes)
            else:
                # Tracking mode: use state management and credence counting
                # Get normalized embedding for tracking
                from .face_utils import normalize_embedding

                normalized_embedding = normalize_embedding(embedding)

                # Use extracted tracking function
                process_tracked_recognition(
                    track_id,
                    db_id,
                    attributes,
                    normalized_embedding,
                    self._face_reid_map,
                    self._credence_count,
                    self._accumulate_embeddings,
                    self._alert_mode,
                    self._alert_once,
                )
