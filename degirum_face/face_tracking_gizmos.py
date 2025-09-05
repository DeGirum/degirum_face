#
# face_tracking_gizmos.py: face tracking gizmo classes implementation
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements supplemental classes and gizmos for face extraction, alignment, and re-identification (reID).
#

import numpy as np
import threading
import copy
from typing import List, Dict, Any, Optional, ClassVar
from dataclasses import dataclass, asdict, field
from enum import Enum

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
from .logging_config import logger
from . import face_utils


@dataclass
class FaceStatus:
    """
    Class to hold detected face runtime status.
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

    def delete(self, expr):
        """
        Delete objects from the map

        Args:
            expr (lambda): logical expression to filter objects to delete
        """
        with self._lock:
            keys_to_delete = [key for key, value in self.map.items() if expr(value)]
            for key in keys_to_delete:
                del self.map[key]


tag_obj_annotate = "object_annotate"  # tag for object annotation meta
tag_face_align = "face_align"  # tag for face alignment and cropping meta
tag_face_search = "face_search"  # tag for face search meta


class ObjectAnnotateGizmo(Gizmo):
    """Object annotating gizmo"""

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_obj_annotate, tag_inference]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return [tag_inference]

    def __init__(
        self,
        object_map: ObjectMap,
        *,
        label_map: dict = {},
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Constructor.

        Args:
            object_map (ObjectMap): The map of object IDs to attributes.
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


@dataclass
class FaceFilterConfig:
    """Configuration for face filtering"""

    # small face filter: face is small when the minimum size of the smaller side of the face bbox
    # is less than threshold in pixels
    enable_small_face_filter: bool = False
    min_face_size: int = 0

    # zone filter: faces is out of zone when the center of the face bbox is outside specified zone polygon
    enable_zone_filter: bool = False
    zone: List[List[int]] = field(default_factory=list)

    # frontal filter: face is frontal when nose keypoint is inside eyes-mouth rectangle
    enable_frontal_filter: bool = False

    # shift filter: face is shifted when all keypoints are grouped in one half of the image
    enable_shift_filter: bool = False

    # reid expiration filter: reid expiration based filtering
    enable_reid_expiration_filter: bool = False
    reid_expiration_frames: int = 10


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
        filters: FaceFilterConfig,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Constructor.

        Args:
            target_image_size (int): Size to which the image should be resized.
            face_reid_map (ObjectMap): The map of face IDs to face attributes; used for filtering. None means no filtering.
            filters (FaceFilterConfig): Configuration for face filtering.
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._image_size = target_image_size
        self._face_reid_map = face_reid_map
        self._filters = filters

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_align, tag_crop]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return [tag_inference, tag_video]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            # get inference result
            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have face detection gizmo in upstream"
                )

            for i, r in enumerate(result.results):

                landmarks = r.get("landmarks")
                if not landmarks or len(landmarks) != 5:
                    logger.info(f"#{i}: skipping reID: invalid landmarks")
                    continue

                keypoints = [np.array(lm["landmark"]) for lm in landmarks]

                # apply filtering based on the face size
                if (
                    self._filters.enable_small_face_filter
                    and self._filters.min_face_size > 0
                ):
                    bbox = r.get("bbox")
                    if bbox is not None:
                        w, h = abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])
                        if min(w, h) < self._filters.min_face_size:
                            logger.info(f"#{i}: skipping reID: face is too small")
                            continue  # skip if the face is too small

                # apply filtering based on zone
                if self._filters.enable_zone_filter:
                    in_zone = r.get(degirum_tools.ZoneCounter.key_in_zone, False)
                    if not in_zone:
                        # skip if the face is not in the specified zone
                        logger.info(f"#{i}: skipping reID: not in zone")
                        continue

                # apply filtering based on face frontality
                if (
                    self._filters.enable_frontal_filter
                    and not face_utils.face_is_frontal(keypoints)
                ):
                    logger.info(f"#{i}: skipping reID: face is not frontal")
                    continue  # skip if the face is not frontal

                # apply filtering based on face shift
                if self._filters.enable_shift_filter and face_utils.face_is_shifted(
                    r["bbox"], keypoints
                ):
                    logger.info(f"#{i}: skipping reID: face is shifted")
                    continue  # skip if the face is shifted

                # apply filtering based on the face reID map
                if (
                    self._face_reid_map is not None
                    and self._filters.enable_reid_expiration_filter
                ):

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
                            logger.info(
                                f"#{i}: skipping reID for track_id {track_id}, frame {frame_id}: reID not expired"
                            )
                            continue

                        delta = min(
                            self._filters.reid_expiration_frames,
                            2
                            * (
                                face_status.next_reid_frame
                                - face_status.last_reid_frame
                            ),
                        )
                        face_status.last_reid_frame = frame_id
                        face_status.next_reid_frame = frame_id + delta
                    self._face_reid_map.put(track_id, face_status)

                crop_img = face_utils.face_align_and_crop(
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
                logger.info(f"#{i}: sending cropped face, keypoints={keypoints}")
                self.send_result(StreamData(crop_img, new_meta))

            # delete expired faces from the map
            if (
                self._filters.enable_reid_expiration_filter
                and self._face_reid_map is not None
            ):
                self._face_reid_map.delete(
                    lambda x: x.last_reid_frame + self._filters.reid_expiration_frames
                    < frame_id
                )


class AlertMode(Enum):
    """Alert mode for face search gizmo"""

    NONE = 0  # no alert
    ON_UNKNOWNS = 1  # set alert on unknown faces
    ON_KNOWNS = 2  # set alert on known faces
    ON_ALL = 3  # set alert on all detected faces


class FaceSearchGizmo(Gizmo):
    """Face reID search gizmo"""

    # meta keys for face search information
    key_face_db_id = "face_db_id"  # face database ID
    key_face_attributes = "face_attributes"  # face attributes
    key_face_embeddings = "face_embeddings"  # face embeddings
    key_face_similarity_score = (
        "face_similarity_score"  # similarity score of the face embedding
    )

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
    ):
        """
        Constructor.

        Args:
            face_reid_map (ObjectMap): The map of face IDs to face attributes.
            db (ReID_Database): vector database object
            credence_count (int): Number of times the face is recognized before confirming it.
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
            accumulate_embeddings (bool): Whether to accumulate embeddings in the face map.
            alert_mode (AlertMode): Mode of alerting for the face search.
            alert_once (bool): Whether to trigger the alert only once for the given face.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._face_reid_map = face_reid_map
        self._db = db
        self._credence_count = credence_count
        self._accumulate_embeddings = accumulate_embeddings
        self._alert_mode = alert_mode
        self._alert_once = alert_once

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_search]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        tags = [tag_inference]
        if self._face_reid_map is not None:
            tags.append(tag_face_align)
        return tags

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
            db_id, attributes, score = self._db.get_attributes_by_embedding(embedding)

            # send the result to the output
            new_meta = data.meta.clone()
            new_meta.append(
                {
                    self.key_face_db_id: db_id,
                    self.key_face_attributes: attributes,
                    self.key_face_embeddings: embedding,
                    self.key_face_similarity_score: score,
                },
                self.get_tags(),
            )
            logger.info(
                f"Face search result: id={db_id}, attr={attributes}, score={score:.2f}"
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
                            logger.info(f"Alert triggered for track_id {track_id}")
                            self._face_reid_map.set_alert(True)
                            face.is_alerted = True

                    self._face_reid_map.put(track_id, face)
