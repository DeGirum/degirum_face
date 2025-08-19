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
    AiObjectDetectionCroppingGizmo,
    tag_inference,
    tag_video,
    tag_crop,
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

    # meta keys (repeated from AiObjectDetectionCroppingGizmo to be compatible with CropCombiningGizmo)
    key_original_result = AiObjectDetectionCroppingGizmo.key_original_result
    key_cropped_result = AiObjectDetectionCroppingGizmo.key_cropped_result
    key_cropped_index = AiObjectDetectionCroppingGizmo.key_cropped_index
    key_is_last_crop = AiObjectDetectionCroppingGizmo.key_is_last_crop

    def __init__(
        self,
        *,
        target_image_size: int,
        face_reid_map: Optional[ObjectMap] = None,
        reid_expiration_frames: int = 0,
        zone_ids: Optional[List[int]] = None,
        min_face_size: int = 0,
        apply_landmark_heuristic_filtering: bool = True,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Initialize face extraction and alignment gizmo with grouped configuration.

        Args:
            target_image_size (int): Size to which the image should be resized.
            face_reid_map (ObjectMap): The map of face IDs to face attributes; used for filtering. None means no filtering.
            reid_expiration_frames (int): Number of frames after which the face reID needs to be repeated.
            zone_ids (List[int]): List of zone IDs to filter the faces. None means no filtering.
            min_face_size (int): Minimum size of the smaller side of the face bbox in pixels to be considered for reID. 0 means no filtering.
            apply_landmark_heuristic_filtering (bool): Whether to apply heuristic filtering based on face landmarks analysis.
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
        """
        super().__init__([(stream_depth, allow_drop)])

        # Validate target image size
        if target_image_size <= 0:
            raise ValueError("target_image_size must be positive")

        self._image_size = target_image_size
        self._face_reid_map = face_reid_map
        self._reid_expiration_frames = reid_expiration_frames
        self._zone_ids = zone_ids
        self._min_face_size = min_face_size
        self._apply_landmark_heuristic_filtering = apply_landmark_heuristic_filtering

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_align, tag_crop]

    def run(self):
        """Run gizmo - clean and simple flow."""
        for data in self.get_input(0):
            if self._abort:
                break

            try:
                # Get input data
                result, frame_id = self._get_input_data(data)

            for i, r in enumerate(result.results):

                landmarks = r.get("landmarks")
                if not landmarks or len(landmarks) != 5:
                    continue

                # apply filtering based on the face size
                if self._min_face_size > 0:
                    bbox = r.get("bbox")
                    if bbox is not None:
                        w, h = abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])
                        if min(w, h) < self._min_face_size:
                            continue  # skip if the face is too small

                # apply filtering based on zone IDs
                if self._zone_ids:
                    in_zone = r.get(degirum_tools.ZoneCounter.key_in_zone)
                    if in_zone is None or all(
                        not in_zone[zid] for zid in self._zone_ids if zid < len(in_zone)
                    ):
                        continue

                # apply filtering based on face landmark location
                if self._apply_landmark_heuristic_filtering:
                    keypoints = [np.array(lm["landmark"]) for lm in landmarks]
                    if not self.face_is_frontal(keypoints) or self.face_is_shifted(
                        r["bbox"], keypoints
                    ):
                        continue  # skip if the face is not frontal or is shifted

                # apply filtering based on the face reID map
                if self._face_reid_map is not None:

                    # get the track ID and skip if not available
                    track_id = r.get("track_id")
                    if track_id is None:
                        # no track ID - skip reID
                        continue

                    # get current frame ID
                    video_meta = data.meta.find_last(tag_video)
                    if video_meta is None:
                        raise Exception(
                            f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream"
                        )
                    frame_id = video_meta[VideoSourceGizmo.key_frame_id]

                    face_status = self._face_reid_map.get(track_id)
                    if face_status is None:
                        # new face
                        face_status = FaceStatus(
                            attributes=None,
                            track_id=track_id,
                            last_reid_frame=frame_id,
                            next_reid_frame=frame_id + 1,
                        )
                    else:
                        if frame_id < face_status.next_reid_frame:
                            # skip reID if the face is already in the map and not expired
                            continue

                        delta = min(
                            self._reid_expiration_frames,
                            2
                            * (
                                face_status.next_reid_frame
                                - face_status.last_reid_frame
                            ),
                        )
                        face_status.last_reid_frame = frame_id
                        face_status.next_reid_frame = frame_id + delta
                    self._face_reid_map.put(track_id, face_status)

                crop_img = FaceExtractGizmo.face_align_and_crop(
                    data.data, keypoints, self._image_size
                )

                crop_obj = copy.deepcopy(r)
                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: crop_obj,
                    self.key_cropped_index: i,
                    self.key_is_last_crop: i == len(result.results) - 1,
                }
                new_meta = data.meta.clone()
                new_meta.remove_last(tag_inference)
                new_meta.append(crop_meta, self.get_tags())
                self.send_result(StreamData(crop_img, new_meta))

            # delete expired faces from the map
            if self._reid_expiration_frames > 0 and self._face_reid_map is not None:
                self._face_reid_map.delete(
                    lambda x: x.last_reid_frame + self._reid_expiration_frames
                    < frame_id
                )

    @staticmethod
    def face_is_frontal(landmarks: list) -> bool:
        """
        Check if the face is frontal based on the landmarks.
        Args:
            landmarks (List[np.ndarray]): List of 5 keypoints (landmarks) as (x, y) coordinates in the following order:
                [left eye, right eye, nose, left mouth, right mouth].

        Returns:
            bool: True if the face is frontal, False otherwise.
        """
        Determine if a track should be processed based on ReID filtering logic.

        Args:
            track_id: The face track ID (guaranteed to be valid when this method is called)
            frame_id: Current frame number

        Returns:
            bool: True if the face is shifted to a side of bbox, False otherwise.
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

    # meta keys for face search information
    key_face_db_id = "face_db_id"  # face database ID
    key_face_attributes = "face_attributes"  # face attributes
    key_face_embeddings = "face_embeddings"  # face embeddings

    def __init__(
        self,
        face_reid_map: Optional[ObjectMap],
        db: ReID_Database,
        *,
        credence_count: int = 1,
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

            # search the database for the face embedding
            embedding = result.results[0].get("data").ravel()
            embedding /= np.linalg.norm(embedding)
            db_id, attributes = self._db.get_attributes_by_embedding(embedding)

            # send the result to the output
            new_meta = data.meta.clone()
            new_meta.append(
                {
                    self.key_face_db_id: db_id,
                    self.key_face_attributes: attributes,
                    self.key_face_embeddings: embedding,
                },
                self.get_tags(),
            )
            self.send_result(StreamData(data.data, new_meta))

            # update the face attributes in the map
            if self._face_reid_map is not None:

                # get face crop result
                crop_meta = data.meta.find_last(tag_face_align)
                if crop_meta is None:
                    raise Exception(
                        f"{self.__class__.__name__}: crop meta not found: you need to have {FaceExtractGizmo.__class__.__name__} in upstream"
                    )

                face_obj = crop_meta.get(FaceExtractGizmo.key_cropped_result)
                assert face_obj

                track_id = face_obj.get("track_id")
                assert track_id

                face = self._face_reid_map.get(track_id)
                if face is not None:
                    # existing face - update the attributes
                    if face.db_id == db_id:
                        face.confirmed_count += 1
                    else:
                        face.confirmed_count = 1
                        # reset frame counter when the face changes status for quick reconfirming
                        face.next_reid_frame = face.last_reid_frame + 1

                    face.is_confirmed = face.confirmed_count >= self._credence_count
                    if face.attributes != attributes and not self._alert_once:
                        face.is_alerted = False  # reset alert if attributes changed
                    face.attributes = attributes
                    face.db_id = db_id
                    if self._accumulate_embeddings:
                        face.embeddings.append(embedding)

                    if face.is_confirmed:
                        if (
                            (
                                self._alert_mode == AlertMode.ON_UNKNOWNS
                                and attributes is None
                                and not face.is_alerted
                            )
                            or (
                                self._alert_mode == AlertMode.ON_KNOWNS
                                and attributes is not None
                                and not face.is_alerted
                            )
                            or (
                                self._alert_mode == AlertMode.ON_ALL
                                and not face.is_alerted
                            )
                        ):
                            self._face_reid_map.set_alert(True)
                            face.is_alerted = True

                    self._face_reid_map.put(track_id, face)
