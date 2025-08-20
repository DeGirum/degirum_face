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
from typing import Any, Dict, Generator, Optional, Tuple, Union


def __dir__():
    return [
        "FaceTracking",
        "AlertMode",
        "ObjectMap",
        "ReID_Database",
        "FaceSearchGizmo",
        "FaceExtractGizmo",
        "ObjectAnnotateGizmo",
    ]


from degirum_tools.streams import (
    Composition,
    Watchdog,
    AiAnalyzerGizmo,
    AiSimpleGizmo,
    FPSStabilizingGizmo,
    GeneratorSourceGizmo,
    SinkGizmo,
    VideoDisplayGizmo,
    VideoSaverGizmo,
    VideoSourceGizmo,
    VideoStreamerGizmo,
    tag_video,
)

from .face_tracking_gizmos import (
    ObjectMap,
    FaceSearchGizmo,
    FaceExtractGizmo,
    ObjectAnnotateGizmo,
    AlertMode,
    tag_face_search,
    tag_face_align,
)
from .reid_database import ReID_Database


class FaceTracking:

    annotated_video_suffix = "_annotated"  # suffix for annotated video clips

    def __init__(
        self,
        *,
        hw_location: str,
        model_zoo_url: str,
        face_detector_model_name: str,
        face_reid_model_name: str,
        token: Optional[str] = None,
        face_detector_model_devices: Optional[list] = None,
        face_reid_model_devices: Optional[list] = None,
        db_filename: str,
        clip_storage_config: Optional[degirum_tools.ObjectStorageConfig] = None,
    ):
        """
        Constructor.

        Args:
            hw_location (str): Hardware location for the inference.
            model_zoo_url (str): URL of the model zoo.
            face_detector_model_name (str): Name of the face detection model in the model zoo.
            face_reid_model_name (str): Name of the face reID model in the model zoo.
            clip_storage_config (ObjectStorageConfig): Configuration for the object storage where video clips are stored.
            db_filename (str): Path to the reID database.
            token (str, optional): cloud API token or None to use the token from environment.
            face_detector_model_devices (Optional[list]): List of device indexes for the face detector model. If None, all devices are used.
            face_reid_model_devices (Optional[list]): List of device indexes for the face reID model. If None, all devices are used.
        """
        self._hw_location = hw_location
        self._model_zoo_url = model_zoo_url
        self._face_detector_model_name = face_detector_model_name
        self._face_detector_model_devices = face_detector_model_devices
        self._face_reid_model_name = face_reid_model_name
        self._face_reid_model_devices = face_reid_model_devices
        self._clip_storage_config = clip_storage_config
        self._token = token

        self.db = ReID_Database(db_filename)

    def _load_models(
        self, zone: Optional[list], reid_expiration_frames: Optional[int]
    ) -> Tuple[dg.model.Model, dg.model.Model]:
        """
        Load the face detection and face reID models from the model zoo.
        """
        zoo = dg.connect(
            self._hw_location,
            self._model_zoo_url,
            self._token,
        )
        face_detect_model = zoo.load_model(self._face_detector_model_name)
        if self._face_detector_model_devices:
            face_detect_model.devices_selected = self._face_detector_model_devices

        face_reid_model = zoo.load_model(
            self._face_reid_model_name, non_blocking_batch_predict=True
        )
        if self._face_reid_model_devices:
            face_reid_model.devices_selected = self._face_reid_model_devices

        analyzers = []
        # face tracker
        if reid_expiration_frames is not None:
            analyzers.append(
                degirum_tools.ObjectTracker(
                    track_thresh=0.35,
                    track_buffer=reid_expiration_frames + 1,
                    match_thresh=0.9999,
                    trail_depth=reid_expiration_frames + 1,
                    anchor_point=degirum_tools.AnchorPoint.CENTER,
                    show_overlay=True,
                    show_only_track_ids=True,
                )
            )

        # in-zone counter for all faces
        if zone:
            analyzers.append(
                degirum_tools.ZoneCounter(
                    [zone],
                    triggering_position=degirum_tools.AnchorPoint.CENTER,
                    show_overlay=True,
                )
            )

        # attach tracker and zone counter analyzers to the face detection model
        degirum_tools.attach_analyzers(face_detect_model, analyzers)

        return face_detect_model, face_reid_model

    def list_clips(self) -> Dict[str, dict]:
        """
        List the video clips in the storage.
        Returns a dictionary where the key is the clip filename and value is the dict of
            video clip file objects (of minio.Object type) associated with that clip
            original video clip: "original" key,
            JSON annotations: "json" key,
            annotated video clip: "annotated" key
        """

        ret: dict = {}
        storage = degirum_tools.ObjectStorage(self._clip_storage_config)
        storage.ensure_bucket_exists()
        for f in storage.list_bucket_contents():
            if f.object_name.endswith(".mp4"):
                if FaceTracking.annotated_video_suffix not in f.object_name:
                    key = f.object_name.replace(".mp4", "")
                    ret.setdefault(key, {})["original"] = f
                else:
                    key = f.object_name.replace(
                        FaceTracking.annotated_video_suffix, ""
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
        storage = degirum_tools.ObjectStorage(self._clip_storage_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, filename)
            storage.download_file_from_object_storage(filename, local_path)
            return open(local_path, "rb").read()

    def remove_clip(self, filename: str):
        """
        Remove the video clip from the storage.

        Args:
            filename (str): The name of the video clip to remove.
        """
        storage = degirum_tools.ObjectStorage(self._clip_storage_config)
        storage.delete_file_from_object_storage(filename)

    def run_clip_analysis(
        self, clip, zone: Optional[list], save_annotated: bool = True
    ) -> ObjectMap:
        """
        Run the face analysis pipeline on a video clip.

        Args:
            clip (minio.Object): The video clip file object.
            zone (list): Zone coordinates for in-zone counting (optional).
            save_annotated (bool): Whether to save the annotated video clip to the storage.

        Returns:
            ObjectMap: The map of face IDs to face objects found in the clip. Each face object includes a table of embeddings.
        """

        with tempfile.TemporaryDirectory() as tmpdir:

            # define the storage and paths
            storage = degirum_tools.ObjectStorage(self._clip_storage_config)
            storage.ensure_bucket_exists()
            dirname = os.path.dirname(clip.object_name)
            filename = os.path.basename(clip.object_name)
            file_stem, file_ext = os.path.splitext(filename)
            out_filename = file_stem + FaceTracking.annotated_video_suffix + file_ext
            out_object_name = ((dirname + "/") if dirname else "") + out_filename
            input_video_local_path = os.path.join(tmpdir, filename)
            output_video_local_path = os.path.join(tmpdir, out_filename)

            # download the clip to local storage
            storage.download_file_from_object_storage(
                clip.object_name, input_video_local_path
            )

            # load models
            face_detect_model, face_reid_model = self._load_models(zone, 10)
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
                zone_ids=[0] if zone else None,
                min_face_size=reid_height // 2,
            )

            # face ReID AI gizmo
            face_reid = AiSimpleGizmo(face_reid_model)

            # face reID search gizmo
            face_search = FaceSearchGizmo(
                face_map, self.db, credence_count=1, accumulate_embeddings=True
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

    def run_tracking_pipeline(
        self,
        video_source,
        *,
        zone: Optional[list] = None,
        clip_duration: int = 100,
        reid_expiration_frames: int = 10,
        credence_count: int = 4,
        alert_mode: AlertMode = AlertMode.ON_UNKNOWNS,
        alert_once: bool = True,
        notification_config: str = degirum_tools.notification_config_console,
        notification_message: str = "{time}: Unknown person detected. Saved video: [{filename}]({url})",
        local_display: bool = True,
        stream_name: str = "Face Tracking",
    ) -> Tuple[Composition, Watchdog]:
        """
        Run the face tracking pipeline on streaming video source.

        Args:
            video_source (Any): Path to the video file, local camera index, or RTSP camera URL.
            zone (list): Zone coordinates for in-zone counting (optional).
            clip_duration (int): Duration of the clip in frames for saving clips.
            reid_expiration_frames (int): Number of frames after which the face reID needs to be repeated.
            credence_count (int): Number of frames to consider a face as known.
            notification_config (str): Apprise configuration string for notifications.
            notification_message (str): Message template for notifications.
            local_display (bool): Whether to display the video locally or by RTSP stream.
            stream_name (str): Window title for local display or URL path for RTSP streaming.

        Returns:
            tuple: A tuple containing:
                - Composition: The pipeline composition object.
                - Watchdog: Watchdog object to monitor the pipeline.
        """

        # load models
        face_detect_model, face_reid_model = self._load_models(
            zone, reid_expiration_frames
        )

        face_detect_model.overlay_line_width = 1
        reid_height = face_reid_model.input_shape[0][1]  # reID model input height
        face_map = ObjectMap()  # object map for face attributes

        #
        # define gizmos
        #

        # video source gizmo
        source = VideoSourceGizmo(video_source, retry_on_error=True)
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
            message=notification_message,
            notification_config=notification_config,
            clip_save=True,
            clip_duration=clip_duration,
            clip_pre_trigger_delay=clip_duration // 2,
            clip_embed_ai_annotations=False,
            clip_target_fps=fps,
            storage_config=self._clip_storage_config,
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
            reid_expiration_frames=reid_expiration_frames,
            zone_ids=[0] if zone else None,
            min_face_size=reid_height // 2,
        )

        # face ReID AI gizmo
        face_reid = AiSimpleGizmo(face_reid_model)

        # face reID search gizmo
        face_search = FaceSearchGizmo(
            face_map,
            self.db,
            credence_count=credence_count,
            alert_mode=alert_mode,
            alert_once=alert_once,
        )

        # display gizmo
        display: Union[VideoDisplayGizmo, VideoStreamerGizmo] = (
            VideoDisplayGizmo(stream_name, show_ai_overlay=True)
            if local_display
            else VideoStreamerGizmo(
                rtsp_url=f"rtsp://localhost:8554/{stream_name}",
                show_ai_overlay=True,
            )
        )

        watchdog = Watchdog(time_limit=20, tps_threshold=1, smoothing=0.95)
        face_detect.watchdog = watchdog  # attach watchdog

        #
        # define pipeline and run it
        #
        composition = Composition(
            source
            >> fps_stabilizer
            >> face_detect
            >> face_annotate
            >> alerts
            >> display,
            face_detect >> face_extract >> face_reid >> face_search,
        )
        composition.start(wait=False)

        return composition, watchdog

    def recognize_batch(
        self, frames: Generator[Any, None, None]
    ) -> Generator[dg.postprocessor.InferenceResults, None, None]:
        """
        Recognize faces in a batch of frames.

        Args:
            frames: A generator yielding frames as numpy arrays or file paths

        Returns:
            A generator yielding inference results provided by face detection model augmented with face recognition results:
                ""
        """

        # load models
        face_detect_model, face_reid_model = self._load_models(None, None)

        reid_height = face_reid_model.input_shape[0][1]  # reID model input height

        # frame source gizmo
        source = GeneratorSourceGizmo(frames)

        # face detector AI gizmo
        face_detect = AiSimpleGizmo(face_detect_model)

        # face crop gizmo
        face_extract = FaceExtractGizmo(
            target_image_size=reid_height,
            face_reid_map=None,
            reid_expiration_frames=1,
            zone_ids=None,
            min_face_size=reid_height // 2,
        )

        # face ReID AI gizmo
        face_reid = AiSimpleGizmo(face_reid_model)

        # face reID search gizmo
        face_search = FaceSearchGizmo(None, self.db)

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
                    ret._frame_info = video_meta[GeneratorSourceGizmo.key_file_path]

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
