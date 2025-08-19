"""
DeGirum Face Tracking Application Package

A comprehensive face tracking and recognition system built on the DeGirum AI platform.
Provides real-time face detection, recognition, and monitoring capabilities with
advanced filtering, database management, and video analytics.

Key Features:
    - Real-time face detection and tracking across video streams
    - Face recognition with ReID (Re-Identification) capabilities
    - Face embedding generation for recognition and verification
    - Comprehensive face filtering system (size, orientation, zones)
    - Persistent face database with embeddings storage
    - Video clip extraction and storage for detected faces
    - Multi-device model distribution for scalable performance
    - Zone-based monitoring with spatial filtering

Core Components:
    - FaceTracking: Main orchestrator class for face tracking pipelines
    - FaceDetector: Clean API for face detection with auto model selection
    - FaceEmbedder: Face embedding generation and verification API
    - Face ReID: Advanced re-identification using facial embeddings
    - Video Processing: Stream-based gizmo architecture for real-time processing
    - Database Management: Persistent storage of face embeddings and metadata

Architecture:
    The system follows a streaming gizmo pattern where each component processes
    video frames in a pipeline. Key gizmos include:

    - VideoSourceGizmo: Captures video from cameras/files
    - FaceExtractGizmo: Detects and extracts faces from frames
    - FaceSearchGizmo: Performs face recognition against database
    - ObjectAnnotateGizmo: Adds visual annotations to video
    - VideoSaverGizmo: Saves annotated video clips

Usage Examples:
    >>> # Face Detection
    >>> detector = FaceDetector("hailo8")
    >>> results = detector.detect("image.jpg")
    >>>
    >>> # Face Embedding
    >>> embedder = FaceEmbedder("hailo8")
    >>> embedding = embedder.embed("face.jpg")
    >>> verification = embedder.verify_faces("person1.jpg", "person2.jpg")
    >>>
    >>> # Initialize face tracking system
    >>> tracker = FaceTracking(
    ...     hw_location="@cloud",
    ...     model_zoo_url="degirum/public",
    ...     face_detector_model_name="yolo_v8n_face_det",
    ...     face_reid_model_name="arcface_resnet50",
    ...     clip_storage_config=storage_config,
    ...     db_filename="faces.db"
    ... )
    >>>
    >>> # Start tracking from video source
    >>> composition = tracker.track_faces_from_video(
    ...     input_video="camera_feed.mp4",
    ...     output_dir="./results"
    ... )

Performance Considerations:
    - Supports multi-GPU deployment for high-throughput scenarios
    - Configurable model distribution across available devices
    - Optimized gizmo pipeline minimizes latency and memory usage
    - Face filtering reduces computational load by early rejection

Integration Points:
    - DeGirum model zoo for AI model management
    - Object storage systems for video clip archival
    - External databases for face metadata and analytics
    - REST APIs for real-time monitoring and control

Copyright DeGirum Corp. 2025
"""

import os
import tempfile
import degirum as dg
import degirum_tools
from typing import Optional, Tuple, Union

from degirum_tools.streams import (
    Composition,
    Watchdog,
    AiAnalyzerGizmo,
    AiSimpleGizmo,
    FPSStabilizingGizmo,
    VideoDisplayGizmo,
    VideoSaverGizmo,
    VideoSourceGizmo,
    VideoStreamerGizmo,
)

from .face_tracking_gizmos import (
    FaceSearchGizmo,
    FaceExtractGizmo,
    ObjectAnnotateGizmo,
)
from .face_data import FaceStatus, AlertMode
from .face_tracking_utils import ObjectMap, TrackFilter
from .face_filters import FaceFilterConfig
from .reid_database import ReID_Database
from .face_detector import FaceDetector, detect_faces
from .face_embedder import FaceEmbedder, embed_face, verify_faces
from .face_recognition import (
    FaceRecognition,
    EnrollmentResult,
    RecognitionResult,
    FaceQualityMetrics,
)
from .pipeline_config import PipelineModelConfig, ModelSpec
from .model_config import get_model_config


# Package-level convenience functions for model and hardware discovery
def get_supported_hardware() -> list:
    """
    Get list of all supported hardware devices across all tasks.

    Returns:
        List of hardware device names (e.g., ["hailo8", "degirum_orca", "cpu"])

    Example:
        >>> import degirum_face
        >>> hardware_list = degirum_face.get_supported_hardware()
        >>> print(f"Supported hardware: {hardware_list}")
    """
    config = get_model_config()
    return config.get_all_hardware()


def get_supported_tasks() -> list:
    """
    Get list of all supported AI tasks.

    Returns:
        List of task names (e.g., ["face_detection", "face_recognition"])

    Example:
        >>> import degirum_face
        >>> tasks = degirum_face.get_supported_tasks()
        >>> print(f"Available tasks: {tasks}")
    """
    config = get_model_config()
    return config.get_all_tasks()


def get_hardware_for_task(task: str) -> list:
    """
    Get list of hardware devices that support a specific task.

    Args:
        task: Task name (e.g., "face_detection", "face_recognition")

    Returns:
        List of hardware device names that support the task

    Example:
        >>> import degirum_face
        >>> hardware = degirum_face.get_hardware_for_task("face_recognition")
        >>> print(f"Hardware supporting face recognition: {hardware}")
    """
    config = get_model_config()
    return config.get_hardware_for_task(task)


def get_tasks_for_hardware(hardware: str) -> list:
    """
    Get list of tasks supported by a specific hardware device.

    Args:
        hardware: Hardware device name (e.g., "hailo8", "degirum_orca")

    Returns:
        List of task names supported by the hardware

    Example:
        >>> import degirum_face
        >>> tasks = degirum_face.get_tasks_for_hardware("hailo8")
        >>> print(f"Tasks supported by hailo8: {tasks}")
    """
    config = get_model_config()
    return config.get_tasks_for_hardware(hardware)


def get_available_models(task: str = None, hardware: str = None) -> list:
    """
    Get list of available models, optionally filtered by task and/or hardware.

    Args:
        task: Optional task filter (e.g., "face_detection", "face_recognition")
        hardware: Optional hardware filter (e.g., "hailo8", "degirum_orca")

    Returns:
        List of model names matching the filters

    Examples:
        >>> import degirum_face
        >>> # All models
        >>> all_models = degirum_face.get_available_models()
        >>>
        >>> # Face detection models only
        >>> detection_models = degirum_face.get_available_models(task="face_detection")
        >>>
        >>> # Hailo8 models only
        >>> hailo8_models = degirum_face.get_available_models(hardware="hailo8")
        >>>
        >>> # Face recognition models for Hailo8
        >>> models = degirum_face.get_available_models(task="face_recognition", hardware="hailo8")
    """
    config = get_model_config()

    if task and hardware:
        return config.get_models_for_task_and_hardware(task, hardware)
    elif task:
        return config.get_models_for_task(task)
    elif hardware:
        return config.get_models_for_hardware(hardware)
    else:
        return config.get_all_models()


def get_model_info(model_name: str) -> dict:
    """
    Get detailed information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model metadata (description, task, hardware, etc.)

    Example:
        >>> import degirum_face
        >>> info = degirum_face.get_model_info("arcface_mobilefacenet--112x112_quant_hailort_hailo8_1")
        >>> print(f"Model description: {info['description']}")
        >>> print(f"Task: {info['task']}")
        >>> print(f"Hardware: {info['hardware']}")
    """
    config = get_model_config()
    return config.get_model_info(model_name)


def get_default_model(hardware: str, task: str) -> str:
    """
    Get the default model for a specific hardware and task combination.

    Args:
        hardware: Hardware device name (e.g., "hailo8", "degirum_orca")
        task: Task name (e.g., "face_detection", "face_recognition")

    Returns:
        Default model name for the hardware/task combination

    Example:
        >>> import degirum_face
        >>> default = degirum_face.get_default_model("hailo8", "face_recognition")
        >>> print(f"Default face recognition model for hailo8: {default}")
    """
    config = get_model_config()
    return config.get_default_model(hardware, task)


def validate_hardware_task_combination(hardware: str, task: str) -> bool:
    """
    Check if a hardware device supports a specific task.

    Args:
        hardware: Hardware device name
        task: Task name

    Returns:
        True if the hardware supports the task, False otherwise

    Example:
        >>> import degirum_face
        >>> is_supported = degirum_face.validate_hardware_task_combination("hailo8", "face_recognition")
        >>> print(f"Hailo8 supports face recognition: {is_supported}")
    """
    config = get_model_config()
    return config.validate_hardware_task_combination(hardware, task)


class FaceTracking:
    """
    Main orchestrator class for comprehensive face tracking and recognition systems.

    FaceTracking provides a high-level interface for building end-to-end face
    monitoring applications. It integrates face detection, recognition, filtering,
    and video analytics into a unified streaming pipeline.

    Core Capabilities:
        - Real-time face detection using DeGirum AI models
        - Face re-identification (ReID) with persistent database storage
        - Advanced face filtering (size, orientation, spatial zones)
        - Automated video clip extraction for detected faces
        - Multi-device model deployment for scalable performance
        - Comprehensive video analytics and annotation

    Architecture Overview:
        The system uses a gizmo-based streaming architecture where each
        component processes video frames in a pipeline:

        Video Source → Face Detection → Face Filtering → Face Recognition
                                ↓
        Video Output ← Annotation ← Database Lookup ← Feature Extraction

    Key Features:
        - Persistent face database with embedding vectors
        - Configurable face quality filtering
        - Zone-based spatial monitoring
        - Automatic video clip generation
        - Multi-model load balancing
        - Real-time performance optimization

    Usage Patterns:
        1. Video File Processing: Analyze pre-recorded video files
        2. Live Camera Monitoring: Real-time face tracking from cameras
        3. Batch Analysis: Process multiple video sources efficiently
        4. Interactive Applications: Integration with UI/control systems

    Performance Characteristics:
        - Scalable across multiple GPU devices
        - Optimized for real-time processing (>30 FPS)
        - Memory-efficient streaming pipeline
        - Configurable quality vs. speed trade-offs

    Database Integration:
        - SQLite-based face embedding storage
        - Automatic database creation and management
        - Support for face enrollment and lookup
        - Persistent face identity tracking

    Example:
        >>> # Initialize face tracking system
        >>> tracker = FaceTracking(
        ...     hw_location="@cloud",
        ...     model_zoo_url="degirum/public",
        ...     face_detector_model_name="yolo_v8n_face_det",
        ...     face_reid_model_name="arcface_resnet50",
        ...     clip_storage_config=storage_config,
        ...     db_filename="employee_faces.db"
        ... )
        >>>
        >>> # Process security camera feed
        >>> composition = tracker.track_faces_from_video(
        ...     input_video="security_cam.mp4",
        ...     output_dir="./security_alerts",
        ...     zone=[{"type": "rectangle", "coords": [100, 100, 500, 400]}]
        ... )
        >>> composition.start()

    Thread Safety:
        The FaceTracking class is designed for single-threaded initialization
        but supports multi-threaded pipeline execution through the gizmo framework.
    """

    annotated_video_suffix = "_annotated"  # suffix for annotated video clips

    def __init__(
        self,
        *,
        hw_location: str,
        model_zoo_url: str,
        face_detector_model_name: str,
        face_reid_model_name: str,
        clip_storage_config: degirum_tools.ObjectStorageConfig,
        db_filename: str,
        token: Optional[str] = None,
        face_detector_model_devices: Optional[list] = None,
        face_reid_model_devices: Optional[list] = None,
    ):
        """
        Initialize the FaceTracking system with AI models and configuration.

        Sets up the core components needed for face detection, recognition,
        and video processing. Initializes database connections and validates
        model accessibility.

        Args:
            hw_location: Hardware deployment target for AI inference:
                - "@cloud": DeGirum cloud infrastructure
                - "@local": Local hardware deployment
                - Specific device specifications (e.g., "cuda:0")

            model_zoo_url: Base URL for accessing AI model repository:
                - "degirum/public": Public DeGirum model zoo
                - Custom model zoo URLs for private deployments

            face_detector_model_name: Name of face detection model in zoo:
                - "yolo_v8n_face_det": Fast YOLO-based face detector
                - "retinaface_resnet50": High-accuracy RetinaFace detector
                - Custom face detection model names

            face_reid_model_name: Name of face recognition model in zoo:
                - "arcface_resnet50": Standard ArcFace embedding model
                - "facenet_inception": Alternative face embedding model
                - Custom face recognition model names

            clip_storage_config: Configuration for video clip storage system:
                - Local filesystem storage configuration
                - Cloud storage (S3, Azure, GCP) configuration
                - Defines where extracted face clips are saved

            db_filename: Path to SQLite database for face storage:
                - Local file path for face embedding database
                - Automatically created if doesn't exist
                - Stores face embeddings and metadata

            token: Authentication token for cloud services (optional):
                - DeGirum cloud API token for model access
                - None to use environment variable token
                - Required for cloud-based deployments

            face_detector_model_devices: Device distribution for face detection:
                - List of device IDs [0, 1, 2] for multi-GPU deployment
                - None to use all available devices automatically
                - Enables load balancing across hardware

            face_reid_model_devices: Device distribution for face recognition:
                - List of device IDs for recognition model deployment
                - Independent of detector device configuration
                - Allows different scaling strategies per model

        Raises:
            ConnectionError: If model zoo is unreachable
            FileNotFoundError: If database directory doesn't exist
            ValueError: If model names are invalid or not found
            AuthenticationError: If token is invalid for cloud access

        Performance Notes:
            - Model loading is deferred until first pipeline creation
            - Database connection is established immediately
            - Device validation occurs during model loading
            - Multi-device setup improves throughput linearly

        Example Configuration:
            >>> # Cloud deployment with multi-GPU
            >>> tracker = FaceTracking(
            ...     hw_location="@cloud",
            ...     model_zoo_url="degirum/public",
            ...     face_detector_model_name="yolo_v8n_face_det",
            ...     face_reid_model_name="arcface_resnet50",
            ...     clip_storage_config=local_storage_config,
            ...     db_filename="./data/faces.db",
            ...     token="your_api_token",
            ...     face_detector_model_devices=[0, 1],
            ...     face_reid_model_devices=[2, 3]
            ... )
        """
        self._hw_location = hw_location
        self._model_zoo_url = model_zoo_url
        self._face_detector_model_name = face_detector_model_name
        self._face_detector_model_devices = face_detector_model_devices
        self._face_reid_model_name = face_reid_model_name
        self._face_reid_model_devices = face_reid_model_devices
        self._clip_storage_config = clip_storage_config
        self._db_filename = db_filename
        self._token = token
        self.db: Optional[ReID_Database] = None
        self._open_db()

    def _open_db(self):
        """
        Open the database for face reID.
        If the database does not exist, create it.
        """
        if self.db is None:
            self.db = ReID_Database(self._db_filename)
        return self.db

    def _load_models(
        self, zone: Optional[list], reid_expiration_frames: int
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

    def list_clips(self):
        """
        List the video clips in the storage.
        Returns a dictionary where the key is the clip filename and value is the list of
            video clip file objects (of minio.Object type) associated with that clip (original video clip, JSON annotations, annotated video clip)
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
                filter_config=FaceFilterConfig(
                    zone_ids=[0] if zone else None,
                    min_face_size=reid_height // 2,
                    enable_frontal_filter=False,
                    enable_shift_filter=False,
                ),
                recognition_only=False,
                reid_config={"object_map": face_map, "expiration_frames": 0},
            )

            # face ReID AI gizmo
            face_reid = AiSimpleGizmo(face_reid_model)

            # face reID search gizmo
            face_search = FaceSearchGizmo(
                face_map, self._open_db(), credence_count=1, accumulate_embeddings=True
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
            filter_config=FaceFilterConfig(
                zone_ids=[0] if zone else None,
                min_face_size=reid_height // 2,
                enable_frontal_filter=False,
                enable_shift_filter=False,
            ),
            recognition_only=False,
            reid_config={
                "object_map": face_map,
                "expiration_frames": reid_expiration_frames,
            },
        )

        # face ReID AI gizmo
        face_reid = AiSimpleGizmo(face_reid_model)

        # face reID search gizmo
        face_search = FaceSearchGizmo(
            face_map,
            self._open_db(),
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
