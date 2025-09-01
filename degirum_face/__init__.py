#
# Face tracking application package
# Copyright DeGirum Corp. 2025
#
# Implements various classes and functions for face tracking application development
#

import os
import tempfile
import degirum as dg
import degirum_tools
from typing import Any, Dict, Iterator, Optional, Tuple, Union


def __dir__():
    return [
        "model_registry",
        "FaceRecognitionConfig",
        "FaceAnnotationConfig",
        "FaceTrackingConfig",
        "FaceRecognition",
        "FaceAnnotation",
        "start_face_tracking_pipeline",
        "ObjectMap",
        "ReID_Database",
        "FaceSearchGizmo",
        "FaceExtractGizmo",
        "ObjectAnnotateGizmo",
        "configure_logging",
        "set_log_level",
        "logging_disable",
    ]


from degirum_tools.streams import (
    Composition,
    Watchdog,
    AiAnalyzerGizmo,
    AiSimpleGizmo,
    FPSStabilizingGizmo,
    IteratorSourceGizmo,
    SinkGizmo,
    VideoDisplayGizmo,
    VideoSaverGizmo,
    VideoSourceGizmo,
    VideoStreamerGizmo,
    tag_inference,
    tag_video,
)

from .reid_database import ReID_Database

from .face_tracking_gizmos import (
    ObjectMap,
    FaceSearchGizmo,
    FaceExtractGizmo,
    ObjectAnnotateGizmo,
    tag_face_search,
    tag_face_align,
)

from .configs import (
    model_registry,
    FaceRecognitionConfig,
    FaceAnnotationConfig,
    FaceTrackingConfig,
)

from .logging_config import set_log_level, logging_disable, logger


def _load_models(
    config: FaceRecognitionConfig,
) -> Tuple[dg.model.Model, dg.model.Model]:
    """
    Load the face detection and face reID models from the model zoo(s).

    Args:
        config (FaceRecognitionConfig): Configuration containing model specifications.

    Returns:
        Tuple[dg.model.Model, dg.model.Model]: Loaded face detection and face reID model objects.
    """

    zoo = config.face_detector_model.zoo_connect()
    face_detect_model = config.face_detector_model.load_model(zoo)
    if config.face_reid_model.zoo_url != config.face_detector_model.zoo_url:
        zoo = config.face_reid_model.zoo_url.zoo_connect()
    face_reid_model = config.face_reid_model.load_model(zoo)
    logger.info(
        f"Loaded models: {config.face_detector_model.model_name}, {config.face_reid_model.model_name}"
    )
    return face_detect_model, face_reid_model


class FaceRecognition:
    """
    Face recognition class for processing images and recognizing faces.
    Provides basic face recognition capabilities: face recognition and face enrolling.
    """

    def __init__(self, config: FaceRecognitionConfig):
        """
        Constructor.

        Args:
            config (FaceRecognitionConfig): Configuration for face recognition.
        """
        assert isinstance(config, FaceRecognitionConfig)
        self.config = config
        self.face_detect_model, self.face_reid_model = _load_models(config)

    def enroll_batch(
        self, frames: Iterator[Any], attributes: Iterator[Any]
    ):
        """
        Enroll a batch of frames for face recognition.

        Args:
            frames (Iterator[Any]): An iterator yielding frames as numpy arrays or file paths
            attributes (Iterator[Any]): An iterator yielding attributes for each frame
        """

        config = self.config
        reid_height = self.face_reid_model.input_shape[0][1]  # reID model input height

        # frame source gizmo
        source = IteratorSourceGizmo(frames)

        # face detector AI gizmo
        face_detect = AiSimpleGizmo(self.face_detect_model)

        # face crop gizmo (no filters)
        face_extract = FaceExtractGizmo(target_image_size=reid_height)

        # face ReID AI gizmo
        face_reid = AiSimpleGizmo(self.face_reid_model)

        # face reID search gizmo
        face_search = FaceSearchGizmo(None, config.db)

        # sink gizmo to collect results
        sink = SinkGizmo()

        with Composition(
            source >> face_detect >> face_extract >> face_reid >> face_search >> sink
        ):
            max_bbox_area = -1
            for r in sink():
                search_meta = r.meta.find_last(tag_face_search)
                crop_meta = r.meta.find_last(tag_face_align)
                if crop_meta[FaceExtractGizmo.key_cropped_index] == 0:
                    attr = next(attributes)
                    max_bbox_area = -1

                bbox = crop_meta[FaceExtractGizmo.key_cropped_result]["bbox"]
                bbox_area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                if bbox_area > max_bbox_area:
                    max_bbox_area = bbox_area
                    embeddings = search_meta[FaceSearchGizmo.key_face_embeddings]

                if crop_meta[FaceExtractGizmo.key_is_last_crop]:
                    cnt, obj_id = config.db.add_embeddings_for_attributes(
                        attr, [embeddings]
                    )
                    logger.info(f"Enrolled: {attr}, {cnt} embeddings, id={obj_id}")

    def enroll_image(self, frame: Any, attributes: Any):
        """
        Enroll a single image for face recognition.

        Args:
            frame (Any): The input image frame as a numpy array or file path.
            attributes (Any): The attributes for the image (e.g., person name).
        """
        self.enroll_batch(iter((frame,)), iter((attributes,)))

    def recognize_batch(
        self, frames: Iterator[Any]
    ) -> Iterator[dg.postprocessor.InferenceResults]:
        """
        Recognize faces in a batch of frames.

        Args:
            frames: An iterator yielding frames as numpy arrays or file paths
            config (FaceRecognitionConfig): Configuration for face recognition.

        Returns:
            An iterator yielding face detection inference results provided by face detection model
            augmented with face recognition results. For each detected object the following key are
            added to the object dictionary:
                "face_embeddings": face embedding vector
                "face_db_id": database ID string of recognized face
                "face_attributes": recognized face attributes (usually person name string)
                "face_similarity_score": face similarity score from the database search
        """

        config = self.config
        reid_height = self.face_reid_model.input_shape[0][1]  # reID model input height

        # frame source gizmo
        source = IteratorSourceGizmo(frames)

        # face detector AI gizmo
        face_detect = AiSimpleGizmo(self.face_detect_model)

        # face crop gizmo (no filters)
        face_extract = FaceExtractGizmo(target_image_size=reid_height)

        # face ReID AI gizmo
        face_reid = AiSimpleGizmo(self.face_reid_model)

        # face reID search gizmo
        face_search = FaceSearchGizmo(None, config.db)

        # sink gizmo to collect results
        sink = SinkGizmo()

        with Composition(
            source >> face_detect >> face_extract >> face_reid >> face_search >> sink
        ):
            ret: Optional[dg.postprocessor.InferenceResults] = None
            for r in sink():
                search_meta = r.meta.find_last(tag_face_search)
                crop_meta = r.meta.find_last(tag_face_align)
                video_meta = r.meta.find_last(tag_video)

                if not ret:
                    ret = crop_meta[FaceExtractGizmo.key_original_result]
                    ret._frame_info = video_meta[IteratorSourceGizmo.key_file_path]

                idx = crop_meta[FaceExtractGizmo.key_cropped_index]
                ret._inference_results[idx]["face_embeddings"] = search_meta[
                    FaceSearchGizmo.key_face_embeddings
                ]
                ret._inference_results[idx]["face_db_id"] = search_meta[
                    FaceSearchGizmo.key_face_db_id
                ]
                ret._inference_results[idx]["face_attributes"] = search_meta[
                    FaceSearchGizmo.key_face_attributes
                ]
                ret._inference_results[idx]["face_similarity_score"] = search_meta[
                    FaceSearchGizmo.key_face_similarity_score
                ]

                if crop_meta[FaceExtractGizmo.key_is_last_crop]:
                    yield ret
                    ret = None

            if ret is not None:
                yield ret

    def recognize_image(self, frame: Any) -> dg.postprocessor.InferenceResults:
        """
        Recognize faces in a single image.

        Note: Use this method for single image recognitions only where throughput is not a concern.
        For efficient pipelined batch processing, use `recognize_batch()`.

        Args:
            frame (Any): The input frame to recognize.

        Returns:
            dg.postprocessor.InferenceResults: The face detection inference results
            augmented with face recognition results. See `recognize_batch()` for more details.
        """

        for ret in self.recognize_batch(iter((frame,))):
            return ret


def _create_analyzers(zone: Optional[list], trail_depth: int) -> list:
    """
    Create analyzers for face tracking.

    Args:
        zone (Optional[list]): Zone coordinates for in-zone counting (optional).
        trail_depth (int): Depth of the tracking trail.

    Returns:
        list: A list of created analyzer instances: [ObjectTracker, ZoneCounter]
        ZoneCounter is added only when zone is defined.
    """

    analyzers = []
    # face tracker
    analyzers.append(
        degirum_tools.ObjectTracker(
            track_thresh=0.35,
            track_buffer=trail_depth,
            match_thresh=0.9999,
            trail_depth=trail_depth,
            anchor_point=degirum_tools.AnchorPoint.CENTER,
            show_overlay=True,
            show_only_track_ids=True,
        )
    )

    # in-zone counter for all faces (if zone is defined)
    if zone:
        analyzers.append(
            degirum_tools.ZoneCounter(
                [zone],
                triggering_position=degirum_tools.AnchorPoint.CENTER,
                show_overlay=True,
            )
        )

    return analyzers


class FaceAnnotation:
    """
    Class to annotate and manage video clips in the object storage.
    """

    annotated_video_suffix = "_annotated"  # suffix for annotated video clips

    def __init__(self, config: FaceAnnotationConfig):
        """
        Constructor.

        Args:
            config (FaceAnnotationConfig): Configuration for face annotation.
        """
        assert isinstance(config, FaceAnnotationConfig)
        self.config = config
        self.storage = degirum_tools.ObjectStorage(config.clip_storage_config)

    def run_clip_annotation(
        self, clip_object_name: str, save_annotated: bool = True
    ) -> ObjectMap:
        """
        Run the face analysis and annotation pipeline on a video clip.

        Args:
            clip_object_name (str): The video clip file object name in object storage.
            save_annotated (bool): Whether to save the annotated video clip to the object storage.

        Returns:
            ObjectMap: The map of face IDs to face objects found in the clip. Each face object includes a table of embeddings.
        """

        config = self.config
        with tempfile.TemporaryDirectory() as tmpdir:

            # define the storage and paths
            storage = degirum_tools.ObjectStorage(config.clip_storage_config)
            storage.ensure_bucket_exists()
            dirname = os.path.dirname(clip_object_name)
            filename = os.path.basename(clip_object_name)
            file_stem, file_ext = os.path.splitext(filename)
            out_filename = file_stem + self.annotated_video_suffix + file_ext
            out_object_name = ((dirname + "/") if dirname else "") + out_filename
            input_video_local_path = os.path.join(tmpdir, filename)
            output_video_local_path = os.path.join(tmpdir, out_filename)

            # download the clip to local storage
            storage.download_file_from_object_storage(
                clip_object_name, input_video_local_path
            )

            # load models
            face_detect_model, face_reid_model = _load_models(config)
            degirum_tools.attach_analyzers(
                face_detect_model, _create_analyzers(config.zone, 10)
            )
            reid_height = face_reid_model.input_shape[0][1]  # reID model input height

            # suppress all annotations
            face_detect_model.overlay_line_width = 0
            face_detect_model.overlay_show_labels = True
            face_detect_model.overlay_show_probabilities = False

            #
            # define gizmos
            #

            # video source gizmo
            source = VideoSourceGizmo(input_video_local_path)

            # face detector AI gizmo
            face_detect = AiSimpleGizmo(face_detect_model)

            face_map = ObjectMap()  # object map for face attributes

            # face crop gizmo
            face_extract = FaceExtractGizmo(
                target_image_size=reid_height,
                face_reid_map=face_map,
                reid_expiration_frames=0,
                zone_ids=[0] if config.zone else None,
                min_face_size=reid_height // 2,
            )

            # face ReID AI gizmo
            face_reid = AiSimpleGizmo(face_reid_model)

            # face reID search gizmo
            face_search = FaceSearchGizmo(
                face_map, config.db, credence_count=1, accumulate_embeddings=True
            )

            # object annotator gizmo
            face_annotate = ObjectAnnotateGizmo(face_map)

            if save_annotated:
                # annotated video saved gizmo
                saver = VideoSaverGizmo(output_video_local_path, show_ai_overlay=True)
                saver.connect_to(face_annotate)

            #
            # define pipeline and run it
            #
            Composition(
                source >> face_detect >> face_extract >> face_reid >> face_search,
                face_detect >> face_annotate,
            ).start()

            # upload the annotated video to the object storage
            if save_annotated:
                storage.upload_file_to_object_storage(
                    output_video_local_path, out_object_name
                )

            # compute K-means clustering on the embeddings
            for id, face in face_map.map.items():
                face.embeddings = degirum_tools.compute_kmeans(face.embeddings)

            return face_map

    def list_clips(self) -> Dict[str, dict]:
        """
        List the video clips in the storage.
        Returns a dictionary where the key is the clip filename and value is the dict of
            video clip file objects (of minio.datatypes.Object type) associated with that clip
            original video clip: "original" key,
            JSON annotations: "json" key,
            annotated video clip: "annotated" key
        """

        ret: dict = {}
        self.storage.ensure_bucket_exists()
        for f in self.storage.list_bucket_contents():
            if f.object_name.endswith(".mp4"):
                if self.annotated_video_suffix not in f.object_name:
                    key = f.object_name.replace(".mp4", "")
                    ret.setdefault(key, {})["original"] = f
                else:
                    key = f.object_name.replace(
                        self.annotated_video_suffix, ""
                    ).replace(".mp4", "")
                    ret.setdefault(key, {})["annotated"] = f
            elif f.object_name.endswith(".json"):
                key = f.object_name.replace(".json", "")
                ret.setdefault(key, {})["json"] = f
            else:
                continue

        return ret

    def download_clip(self, filename: str) -> bytes:
        """
        Download the video clip from the storage.

        Args:
            filename (str): The name of the video clip to download.

        Returns:
            bytes: The bytes of the downloaded video clip.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, filename)
            self.storage.download_file_from_object_storage(filename, local_path)
            return open(local_path, "rb").read()

    def remove_clip(self, filename: str):
        """
        Remove the video clip from the storage.

        Args:
            filename (str): The name of the video clip to remove.
        """
        self.storage.delete_file_from_object_storage(filename)


def start_face_tracking_pipeline(
    config: FaceTrackingConfig,
) -> Tuple[Composition, Watchdog]:
    """
    Run the face tracking pipeline on streaming video source.

    Args:
        config (FaceTrackingConfig): Configuration for face tracking.

    Returns:
        tuple: A tuple containing:
            - Composition: The pipeline composition object.
            - Watchdog: Watchdog object to monitor the pipeline.
    """

    assert isinstance(config, FaceTrackingConfig)

    # load models
    face_detect_model, face_reid_model = _load_models(config)
    degirum_tools.attach_analyzers(
        face_detect_model,
        _create_analyzers(config.zone, config.reid_expiration_frames + 1),
    )

    face_detect_model.overlay_line_width = 1
    reid_height = face_reid_model.input_shape[0][1]  # reID model input height
    face_map = ObjectMap()  # object map for face attributes

    #
    # define gizmos
    #

    # video source gizmo
    source = VideoSourceGizmo(config.video_source, retry_on_error=True)
    _, _, fps = source.get_video_properties()

    # gizmo to keep FPS (to deal with camera disconnects)
    fps_stabilizer = FPSStabilizingGizmo()

    # face detector AI gizmo
    face_detect = AiSimpleGizmo(face_detect_model)

    # object annotator gizmo
    face_annotate = ObjectAnnotateGizmo(face_map)

    # "unknown face" event detector
    unknown_face_event_name = "unknown_face"  # name of the event to be generated
    unknown_face_event_detector = degirum_tools.EventDetector(
        f"""
        Trigger: {unknown_face_event_name}
        when: CustomMetric
        is greater than: 0
        during: [1, frame]
        """,
        custom_metric=lambda result, params: int(face_map.read_alert()),
        show_overlay=False,
    )

    # notifier for unknown face events
    unknown_face_notifier = degirum_tools.EventNotifier(
        "Unknown person detected",
        unknown_face_event_name,
        message=config.notification_message,
        notification_config=config.notification_config,
        clip_save=True,
        clip_duration=config.clip_duration,
        clip_pre_trigger_delay=config.clip_duration // 2,
        clip_embed_ai_annotations=False,
        clip_target_fps=fps,
        storage_config=config.clip_storage_config,
        show_overlay=False,
    )

    # gizmo to execute a chain of analyzers which count unknown faces and generate events and alerts
    alerts = AiAnalyzerGizmo(
        [
            unknown_face_event_detector,
            unknown_face_notifier,
        ]
    )

    # face crop gizmo
    face_extract = FaceExtractGizmo(
        target_image_size=reid_height,
        face_reid_map=face_map,
        reid_expiration_frames=config.reid_expiration_frames,
        zone_ids=[0] if config.zone else None,
        min_face_size=reid_height // 2,
    )

    # face ReID AI gizmo
    face_reid = AiSimpleGizmo(face_reid_model)

    # face reID search gizmo
    face_search = FaceSearchGizmo(
        face_map,
        config.db,
        credence_count=config.credence_count,
        alert_mode=config.alert_mode,
        alert_once=config.alert_once,
    )

    # display gizmo
    display: Union[VideoDisplayGizmo, VideoStreamerGizmo, None] = None
    if config.live_stream_mode == "LOCAL":
        display = VideoDisplayGizmo(config.live_stream_rtsp_url, show_ai_overlay=True)
    elif config.live_stream_mode == "WEB":
        display = VideoStreamerGizmo(
            rtsp_url=f"rtsp://localhost:8554/{config.live_stream_rtsp_url}",
            show_ai_overlay=True,
        )

    watchdog = Watchdog(time_limit=20, tps_threshold=1, smoothing=0.95)
    face_detect.watchdog = watchdog  # attach watchdog

    #
    # define pipeline and run it
    #
    composition = Composition(
        source >> fps_stabilizer >> face_detect >> face_annotate >> alerts >> display,
        face_detect >> face_extract >> face_reid >> face_search,
    )
    composition.start(wait=False)

    return composition, watchdog
