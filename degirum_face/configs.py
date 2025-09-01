#
# Configuration classes for face tracking application
# Copyright DeGirum Corp. 2025
#
# Contains dataclass configurations for face recognition, annotation, and tracking
#

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Optional, Union
import degirum_tools
import jsonschema
import yaml
from .reid_database import ReID_Database
from .face_tracking_gizmos import AlertMode


# Model registry of official face tracking models
model_registry = degirum_tools.ModelRegistry(
    config_file=Path(__file__).parent / "models.yaml"
)


@dataclass
class FaceRecognitionConfig:
    """
    Configuration for basic face recognition activities.

    Attributes:
        face_detector_model (degirum_tools.ModelSpec): Face detection model specification.
        face_reid_model (degirum_tools.ModelSpec): Face re-identification model specification.
        db (ReID_Database): Database for face embeddings.
    """

    face_detector_model: degirum_tools.ModelSpec
    face_reid_model: degirum_tools.ModelSpec
    db: ReID_Database

    # Schema keys for FaceRecognitionConfig
    key_face_detector_model = "face_detector"
    key_face_reid_model = "face_embedder"
    key_db = "database"
    key_model_name = "model_name"
    key_hardware = "hardware"
    key_inference_host_address = "inference_host_address"
    key_zoo_url = "model_zoo_url"
    key_token = "token"
    key_devices = "devices"
    key_db_path = "db_path"
    key_threshold = "similarity_threshold"

    schema = f"""
type: object
$defs:
    model_config:
        type: object
        properties:
            {key_model_name}:
                type: string
                description: "Explicit model name. When specified, {key_hardware} key is ignored."
            {key_hardware}:
                type: string
                pattern: "^[A-Z0-9_]+/[A-Z0-9_]+$"
                description: "Hardware to use in a form RUNTIME/DEVICE. Model registry will be queried for models for this hardware."
            {key_inference_host_address}:
                type: string
                description: "Inference host designator: @cloud, @local, or AI server hostname"
            {key_zoo_url}:
                type: string
                description: "Filepath for local model zoos or URL path in a form workspace/zoo for cloud model zoos"
            {key_token}:
                type: string
                description: "Cloud API access token for accessing cloud zoo"
            {key_devices}:
                type: array
                items:
                    type: integer
                description: "List of device IDs to use for inference. Omit to use all available devices."
        required:
            - {key_inference_host_address}
        additionalProperties: false
properties:
    {key_face_detector_model}:
        $ref: "#/$defs/model_config"
    {key_face_reid_model}:
        $ref: "#/$defs/model_config"
    {key_db}:
        type: object
        properties:
            {key_db_path}:
                type: string
            {key_threshold}:
                type: number
                minimum: 0
                maximum: 1
                description: "Cosine distance threshold for considering two face embeddings as similar."
        required:
            - {key_db_path}
        additionalProperties: false
required:
    - {key_face_detector_model}
    - {key_face_reid_model}
    - {key_db}
additionalProperties: true
    """

    @classmethod
    def from_yaml(
        cls, *, yaml_file: Union[str, Path, None] = None, yaml_str: Optional[str] = None
    ):
        """
        Create configuration class instance from YAML configuration.

        Args:
            yaml_file (Optional[str]): Path to the YAML file.
            yaml_str (Optional[str]): YAML configuration as a string.

        Returns:
            Config object created from YAML.
        """

        if yaml_file:
            path = Path(yaml_file)

            if not path.is_absolute() and path.parent == Path("."):
                # get caller’s filename
                caller_file = Path(inspect.stack()[1].filename).resolve()
                if caller_file.exists():
                    caller_dir = caller_file.parent
                    path = caller_dir / path
                else:
                    # fallback: use current working directory
                    path = Path.cwd() / path

            with open(path, "r") as f:
                settings = yaml.safe_load(f)

        elif yaml_str:
            settings = yaml.safe_load(yaml_str)
        else:
            raise ValueError("Either yaml_file or yaml_str must be provided")

        return cls.from_settings(settings)

    @staticmethod
    def from_settings(settings: dict) -> "FaceRecognitionConfig":
        """
        Create FaceRecognitionConfig from settings dictionary.
        Args:
            settings (dict): Configuration settings as loaded from YAML.

        """

        jsonschema.validate(
            instance=settings, schema=yaml.safe_load(FaceRecognitionConfig.schema)
        )

        def get_model_spec(model_settings, task):
            model_name = model_settings.get(FaceRecognitionConfig.key_model_name)
            hw = model_settings.get(FaceRecognitionConfig.key_hardware)

            if model_name:
                model_spec = degirum_tools.ModelSpec(
                    model_name=model_name,
                    zoo_url=model_settings.get(FaceRecognitionConfig.key_zoo_url),
                    token=model_settings.get(FaceRecognitionConfig.key_token),
                    load_kwargs=dict(
                        devices_selected=model_settings.get(
                            FaceRecognitionConfig.key_devices
                        )
                    ),
                )
            elif hw:
                model_spec = (
                    model_registry.for_task(task)
                    .for_hardware(hw)
                    .model_spec(
                        inference_host_address=model_settings[
                            FaceRecognitionConfig.key_inference_host_address
                        ],
                        zoo_url=model_settings.get(FaceRecognitionConfig.key_zoo_url),
                        token=model_settings.get(FaceRecognitionConfig.key_token),
                        load_kwargs=dict(
                            devices_selected=model_settings.get(
                                FaceRecognitionConfig.key_devices
                            )
                        ),
                    )
                )
            else:
                raise ValueError(
                    f"Either {FaceRecognitionConfig.key_model_name} or {FaceRecognitionConfig.key_hardware} must be specified in model configuration for key '{task}' model"
                )
            return model_spec

        face_detector_model = get_model_spec(
            settings[FaceRecognitionConfig.key_face_detector_model], "face_detection"
        )
        face_reid_model = get_model_spec(
            settings[FaceRecognitionConfig.key_face_reid_model], "face_embedding"
        )
        dg_settings = settings[FaceRecognitionConfig.key_db]
        db = ReID_Database(
            db_path=dg_settings.get(FaceRecognitionConfig.key_db_path),
            threshold=dg_settings.get(FaceRecognitionConfig.key_threshold, 0.3),
        )

        return FaceRecognitionConfig(
            face_detector_model=face_detector_model,
            face_reid_model=face_reid_model,
            db=db,
        )


@dataclass
class FaceAnnotationConfig(FaceRecognitionConfig):
    """
    Configuration for face annotation.

    Attributes:
        clip_storage_config (degirum_tools.ObjectStorageConfig): Storage configuration for video clips.
        zone (list): Zone coordinates as array of [x, y] points, minimum 3 points (optional).
        min_face_size_percent (float): Minimum allowed face size as percentage of reID model input size.
        enable_frontal_filter (bool): Enable face frontality filter.
        enable_shift_filter (bool): Enable face shift filter.
    """

    clip_storage_config: degirum_tools.ObjectStorageConfig
    zone: Optional[list] = None
    min_face_size_percent: float = 50.0
    enable_frontal_filter: bool = True
    enable_shift_filter: bool = True

    # Schema keys for FaceAnnotationConfig
    key_storage = "storage"
    key_filters = "face_filters"
    key_attention_zone = "attention_zone"
    key_min_face_size_percent = "min_face_size_percent"
    key_enable_frontal_filter = "enable_frontal_filter"
    key_enable_shift_filter = "enable_shift_filter"
    key_endpoint = "endpoint"
    key_access_key = "access_key"
    key_secret_key = "secret_key"
    key_bucket = "bucket"
    key_url_expiration_s = "url_expiration_s"

    schema = f"""
type: object
properties:
    {key_storage}:
        type: object
        properties:
            {key_endpoint}:
                type: string
            {key_access_key}:
                type: string
            {key_secret_key}:
                type: string
            {key_bucket}:
                type: string
            {key_url_expiration_s}:
                type: integer
                minimum: 1
        required:
            - {key_endpoint}
            - {key_access_key}
            - {key_secret_key}
            - {key_bucket}
        additionalProperties: false
    {key_filters}:
        type: object
        properties:
            {key_attention_zone}:
                type: array
                items:
                    type: array
                    items:
                        type: number
                    minItems: 2
                    maxItems: 2
                minItems: 3
                description: "Zone coordinates as array of [x, y] points, minimum 3 points"
            {key_min_face_size_percent}:
                type: number
                minimum: 0
                maximum: 100
                description: "Minimum allowed face size as percentage of reID model input size"
            {key_enable_frontal_filter}:
                type: boolean
                description: "Enable face frontality filter"
            {key_enable_shift_filter}:
                type: boolean
                description: "Enable face shift filter"
        additionalProperties: false
required:
    - {key_storage}
    - {key_filters}
additionalProperties: true
    """

    @staticmethod
    def from_settings(settings: dict) -> "FaceAnnotationConfig":
        """
        Create FaceAnnotationConfig from settings dictionary.
        Args:
            settings (dict): Configuration settings as loaded from YAML.

        """

        jsonschema.validate(
            instance=settings, schema=yaml.safe_load(FaceAnnotationConfig.schema)
        )

        base_config = FaceRecognitionConfig.from_settings(settings)

        storage_settings = settings[FaceAnnotationConfig.key_storage]
        clip_storage_config = degirum_tools.ObjectStorageConfig(
            endpoint=storage_settings[FaceAnnotationConfig.key_endpoint],
            access_key=storage_settings[FaceAnnotationConfig.key_access_key],
            secret_key=storage_settings[FaceAnnotationConfig.key_secret_key],
            bucket=storage_settings[FaceAnnotationConfig.key_bucket],
            url_expiration_s=storage_settings[
                FaceAnnotationConfig.key_url_expiration_s
            ],
        )

        # Extract filter settings
        filters_settings = settings[FaceAnnotationConfig.key_filters]
        zone = filters_settings.get(FaceAnnotationConfig.key_attention_zone)
        min_face_size_percent = filters_settings.get(
            FaceAnnotationConfig.key_min_face_size_percent, 50.0
        )
        enable_frontal_filter = filters_settings.get(
            FaceAnnotationConfig.key_enable_frontal_filter, True
        )
        enable_shift_filter = filters_settings.get(
            FaceAnnotationConfig.key_enable_shift_filter, True
        )

        return FaceAnnotationConfig(
            **vars(base_config),
            clip_storage_config=clip_storage_config,
            zone=zone,
            min_face_size_percent=min_face_size_percent,
            enable_frontal_filter=enable_frontal_filter,
            enable_shift_filter=enable_shift_filter,
        )


@dataclass
class FaceTrackingConfig(FaceAnnotationConfig):
    """
    Configuration for face tracking.

    Attributes:
        reid_expiration_frames (int): Number of frames after which the face reID needs to be repeated.
        credence_count (int): Number of frames to consider a face confirmed.
        alert_mode (AlertMode): Alert mode configuration.
        alert_once (bool): Whether to trigger the alert only once or each time alert condition happens.
        clip_duration (int): Duration of the clip in frames for saving clips.
        notification_config (str): Apprise configuration string for notifications.
        notification_message (str): Message template for notifications.
        video_source: Video source; can be integer number to use local camera, RTSP URL, or path to video file.
        live_stream_mode (str): Live stream mode: "LOCAL", "WEB", or "NONE".
        live_stream_rtsp_url (str): RTSP URL path for live stream (if mode is "WEB").
    """

    reid_expiration_frames: int = 10
    credence_count: int = 4
    alert_mode: AlertMode = AlertMode.NONE
    alert_once: bool = True
    clip_duration: int = 100
    notification_config: str = degirum_tools.notification_config_console
    notification_message: str = (
        "{time}: Unknown person detected. Saved video: [{filename}]({url})"
    )
    video_source: Union[int, str] = 0
    live_stream_mode: str = "LOCAL"
    live_stream_rtsp_url: str = "face_tracking"

    # Schema keys for FaceTrackingConfig
    key_reid = "reid"
    key_alerts = "alerts"
    key_reid_expiration_frames = "reid_expiration_frames"
    key_credence_count = "credence_count"
    key_alert_mode = "alert_mode"
    key_alert_once = "alert_once"
    key_clip_duration = "clip_duration"
    key_notification_config = "notification_config"
    key_notification_message = "notification_message"
    key_video_source = "video_source"
    key_live_stream = "live_stream"
    key_live_stream_mode = "mode"
    key_live_stream_rtsp_url = "rtsp_url"

    schema = f"""
type: object
properties:
    {key_video_source}:
        oneOf:
            - type: integer
              minimum: 0
              description: "Camera device index (0, 1, 2, etc.)"
            - type: string
              description: "RTSP URL or path to video file"
        description: "Video source; can be integer number to use local camera, RTSP URL, or path to video file"
    {key_live_stream}:
        type: object
        properties:
            {key_live_stream_mode}:
                type: string
                enum: ["LOCAL", "WEB", "NONE"]
                description: "Live stream mode"
            {key_live_stream_rtsp_url}:
                type: string
                description: "RTSP URL path for live stream (if mode is 'WEB')"
        required:
            - {key_live_stream_mode}
        additionalProperties: false
    {key_reid}:
        type: object
        properties:
            {key_reid_expiration_frames}:
                type: integer
                minimum: 1
                description: "Number of frames after which the face reID needs to be repeated"
            {key_credence_count}:
                type: integer
                minimum: 1
                description: "Number of frames to consider a face confirmed"
        additionalProperties: false
    {key_alerts}:
        type: object
        properties:
            {key_alert_mode}:
                type: string
                enum: ["ON_UNKNOWNS", "ON_KNOWNS", "ON_ALL", "NONE"]
                description: "Alert mode configuration"
            {key_alert_once}:
                type: boolean
                description: "Whether to trigger the alert only once or each time alert condition happens"
            {key_clip_duration}:
                type: integer
                minimum: 1
                description: "Duration of the clip in frames for saving clips"
            {key_notification_config}:
                type: string
                description: "Apprise configuration string for notifications"
            {key_notification_message}:
                type: string
                description: "Message template for notifications"
        required:
            - {key_alert_mode}
        additionalProperties: false
required:
    - {key_video_source}
    - {key_live_stream}
    - {key_reid}
    - {key_alerts}
additionalProperties: true
    """

    @staticmethod
    def from_settings(settings: dict) -> "FaceTrackingConfig":
        """
        Create FaceTrackingConfig from settings dictionary.
        Args:
            settings (dict): Configuration settings as loaded from YAML.

        """
        jsonschema.validate(
            instance=settings, schema=yaml.safe_load(FaceTrackingConfig.schema)
        )

        base_config = FaceAnnotationConfig.from_settings(settings)

        # Extract reid settings
        reid_settings = settings[FaceTrackingConfig.key_reid]
        reid_expiration_frames = reid_settings.get(
            FaceTrackingConfig.key_reid_expiration_frames, 10
        )
        credence_count = reid_settings.get(FaceTrackingConfig.key_credence_count, 4)

        # Extract alerts settings
        alerts_settings = settings[FaceTrackingConfig.key_alerts]
        alert_mode_str = alerts_settings.get(
            FaceTrackingConfig.key_alert_mode, "ON_UNKNOWNS"
        )
        alert_mode = AlertMode[alert_mode_str] if alert_mode_str else AlertMode.NONE
        alert_once = alerts_settings.get(FaceTrackingConfig.key_alert_once, True)
        clip_duration = alerts_settings.get(FaceTrackingConfig.key_clip_duration, 100)
        notification_config = alerts_settings.get(
            FaceTrackingConfig.key_notification_config,
            degirum_tools.notification_config_console,
        )
        notification_message = alerts_settings.get(
            FaceTrackingConfig.key_notification_message,
            "{time}: Unknown person detected. Saved video: [{filename}]({url})",
        )

        # Extract video_source
        video_source = settings.get(FaceTrackingConfig.key_video_source, 0)

        # Extract live_stream settings
        live_stream_settings = settings[FaceTrackingConfig.key_live_stream]
        live_stream_mode = live_stream_settings.get(
            FaceTrackingConfig.key_live_stream_mode, "LOCAL"
        )
        live_stream_rtsp_url = live_stream_settings.get(
            FaceTrackingConfig.key_live_stream_rtsp_url, "face_tracking"
        )

        return FaceTrackingConfig(
            **vars(base_config),
            reid_expiration_frames=reid_expiration_frames,
            credence_count=credence_count,
            alert_mode=alert_mode,
            alert_once=alert_once,
            clip_duration=clip_duration,
            notification_config=notification_config,
            notification_message=notification_message,
            video_source=video_source,
            live_stream_mode=live_stream_mode,
            live_stream_rtsp_url=live_stream_rtsp_url,
        )
