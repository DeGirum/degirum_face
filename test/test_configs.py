#
# test_configs.py: Unit tests for configuration classes
# Copyright DeGirum Corp. 2025
#
# Unit tests for FaceRecognitionConfig and related configuration classes
# (AI-generated)
#

import pytest
import tempfile
import os
import yaml
import degirum_tools
from degirum_face.configs import (
    FaceRecognitionConfig,
    FaceClipManagerConfig,
    FaceTrackingConfig,
)
from degirum_face.reid_database import ReID_Database
from degirum_face.face_tracking_gizmos import FaceFilterConfig, AlertMode


def test_config_face_recognition():
    """Test FaceRecognitionConfig class functionality"""

    # Test 1: Basic instantiation with required parameters
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db.lance")

        # Create model specs
        face_detection_model = degirum_tools.ModelSpec(
            model_name="test_face_detector", inference_host_address="@cloud"
        )
        face_embedding_model = degirum_tools.ModelSpec(
            model_name="test_face_embedder", inference_host_address="@cloud"
        )

        # Create database
        db = ReID_Database(db_path=db_path, threshold=0.5)

        # Create face filter config
        face_filter_config = FaceFilterConfig(
            enable_small_face_filter=True, min_face_size=50
        )

        # Test basic instantiation
        config = FaceRecognitionConfig(
            face_detection_model=face_detection_model,
            face_embedding_model=face_embedding_model,
            db=db,
            face_filter_config=face_filter_config,
        )

        assert config.face_detection_model == face_detection_model
        assert config.face_embedding_model == face_embedding_model
        assert config.db == db
        assert config.face_filter_config == face_filter_config

    # Test 2: Default face_filter_config
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db2.lance")
        db = ReID_Database(db_path=db_path)

        config = FaceRecognitionConfig(
            face_detection_model=face_detection_model,
            face_embedding_model=face_embedding_model,
            db=db,
        )

        assert isinstance(config.face_filter_config, FaceFilterConfig)
        assert config.face_filter_config.enable_small_face_filter is False
        assert config.face_filter_config.min_face_size == 0

    # Test 3: from_settings method with valid configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db3.lance")

        settings = {
            "face_detector": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "face_embedder": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "database": {"db_path": db_path, "similarity_threshold": 0.6},
            "face_filters": {
                "enable_small_face_filter": True,
                "min_face_size": 30,
                "enable_zone_filter": False,
                "enable_frontal_filter": True,
                "enable_shift_filter": False,
                "enable_reid_expiration_filter": True,
                "reid_expiration_frames": 20,
            },
        }

        config = FaceRecognitionConfig.from_settings(settings)

        assert (
            config.face_detection_model.model_name
            == "yolov8n_relu6_widerface_kpts--640x640_quant_n2x_orca1_1"
        )
        assert (
            config.face_embedding_model.model_name
            == "arcface_mobilefacenet--112x112_quant_n2x_orca1_1"
        )
        assert config.db._threshold == 0.6
        assert config.face_filter_config.enable_small_face_filter
        assert config.face_filter_config.min_face_size == 30
        assert config.face_filter_config.enable_frontal_filter
        assert config.face_filter_config.enable_reid_expiration_filter
        assert config.face_filter_config.reid_expiration_frames == 20

    # Test 4: from_settings with hardware specification
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db4.lance")

        settings = {
            "face_detector": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "face_embedder": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "database": {"db_path": db_path},
        }

        config = FaceRecognitionConfig.from_settings(settings)

        assert config.face_detection_model is not None
        assert config.face_embedding_model is not None
        assert config.db._threshold == 0.4  # default threshold

    # Test 5: from_yaml method with string
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db5.lance").replace("\\", "/")

        yaml_str = f"""
face_detector:
  hardware: N2X/ORCA1
  inference_host_address: "@cloud"
face_embedder:
  hardware: N2X/ORCA1
  inference_host_address: "@cloud"
database:
  db_path: "{db_path}"
  similarity_threshold: 0.7
face_filters:
  enable_small_face_filter: true
  min_face_size: 25
  enable_zone_filter: true
  zone: [[10, 10], [200, 10], [200, 150], [10, 150]]
  enable_frontal_filter: true
  enable_shift_filter: true
  enable_reid_expiration_filter: true
  reid_expiration_frames: 15

        """

        config, loaded_settings = FaceRecognitionConfig.from_yaml(yaml_str=yaml_str)

        assert (
            config.face_detection_model.model_name
            == "yolov8n_relu6_widerface_kpts--640x640_quant_n2x_orca1_1"
        )
        assert (
            config.face_embedding_model.model_name
            == "arcface_mobilefacenet--112x112_quant_n2x_orca1_1"
        )
        assert config.db._threshold == 0.7
        # Check all 5 face filters are enabled and configured correctly
        assert config.face_filter_config.enable_small_face_filter
        assert config.face_filter_config.min_face_size == 25
        assert config.face_filter_config.enable_zone_filter
        assert config.face_filter_config.zone == [
            [10, 10],
            [200, 10],
            [200, 150],
            [10, 150],
        ]
        assert config.face_filter_config.enable_frontal_filter
        assert config.face_filter_config.enable_shift_filter
        assert config.face_filter_config.enable_reid_expiration_filter
        assert config.face_filter_config.reid_expiration_frames == 15
        assert isinstance(loaded_settings, dict)

    # Test 6: from_yaml method with file
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db6.lance")
        yaml_file_path = os.path.join(temp_dir, "test_config.yaml")

        yaml_content = {
            "face_detector": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "face_embedder": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "database": {"db_path": db_path, "similarity_threshold": 0.8},
        }

        with open(yaml_file_path, "w") as f:
            yaml.dump(yaml_content, f)

        config, loaded_settings = FaceRecognitionConfig.from_yaml(
            yaml_file=yaml_file_path
        )

        assert (
            config.face_detection_model.model_name
            == "yolov8n_relu6_widerface_kpts--640x640_quant_n2x_orca1_1"
        )
        assert (
            config.face_embedding_model.model_name
            == "arcface_mobilefacenet--112x112_quant_n2x_orca1_1"
        )
        assert config.db._threshold == 0.8
        assert isinstance(loaded_settings, dict)

    # Test 7: Error cases

    # Missing required parameters in settings
    with pytest.raises(Exception):
        FaceRecognitionConfig.from_settings({})

    # Missing either yaml_file or yaml_str
    with pytest.raises(
        ValueError, match="Either yaml_file or yaml_str must be provided"
    ):
        FaceRecognitionConfig.from_yaml()

    # Invalid schema - missing required fields
    invalid_settings = {
        "face_detector": {
            "model_name": "test"
            # missing inference_host_address
        }
    }
    with pytest.raises(Exception):
        FaceRecognitionConfig.from_settings(invalid_settings)


def test_config_face_annotation():
    """Test FaceClipManagerConfig class functionality"""

    # Test 1: Basic instantiation with required parameters
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db.lance")

        # Create model specs
        face_detection_model = degirum_tools.ModelSpec(
            model_name="test_face_detector",
            inference_host_address="@cloud",
            zoo_url="@cloud",
        )
        face_embedding_model = degirum_tools.ModelSpec(
            model_name="test_face_embedder",
            inference_host_address="@cloud",
            zoo_url="@cloud",
        )

        # Create database
        db = ReID_Database(db_path=db_path, threshold=0.5)

        # Create storage config
        clip_storage_config = degirum_tools.ObjectStorageConfig(
            endpoint="http://localhost:9000",
            access_key="testkey",
            secret_key="testsecret",
            bucket="test_bucket",
        )

        # Test basic instantiation
        config = FaceClipManagerConfig(
            face_detection_model=face_detection_model,
            face_embedding_model=face_embedding_model,
            db=db,
            clip_storage_config=clip_storage_config,
        )

        # Verify inherited attributes
        assert config.face_detection_model == face_detection_model
        assert config.face_embedding_model == face_embedding_model
        assert config.db == db

        # Verify FaceClipManagerConfig specific attributes
        assert config.clip_storage_config == clip_storage_config
        assert config.clip_storage_config.endpoint == "http://localhost:9000"
        assert config.clip_storage_config.access_key == "testkey"
        assert config.clip_storage_config.secret_key == "testsecret"
        assert config.clip_storage_config.bucket == "test_bucket"

    # Test 2: Default clip_storage_config
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db2.lance")
        db = ReID_Database(db_path=db_path)

        config = FaceClipManagerConfig(
            face_detection_model=face_detection_model,
            face_embedding_model=face_embedding_model,
            db=db,
        )

        # Verify default storage config
        assert isinstance(config.clip_storage_config, degirum_tools.ObjectStorageConfig)
        assert config.clip_storage_config.endpoint == "./"
        assert config.clip_storage_config.access_key == ""
        assert config.clip_storage_config.secret_key == ""
        assert config.clip_storage_config.bucket == "face_clips"

    # Test 3: from_settings method with storage configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db3.lance")

        settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path, "similarity_threshold": 0.6},
            "storage": {
                "endpoint": "s3.amazonaws.com",
                "access_key": "AKIAIOSFODNN7EXAMPLE",
                "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "bucket": "my-face-clips",
                "url_expiration_s": 3600,
            },
        }

        config = FaceClipManagerConfig.from_settings(settings)

        # Verify inherited functionality works
        assert config.face_detection_model.model_name == "test_detector"
        assert config.face_embedding_model.model_name == "test_embedder"
        assert config.db._threshold == 0.6

        # Verify storage configuration
        assert config.clip_storage_config.endpoint == "s3.amazonaws.com"
        assert config.clip_storage_config.access_key == "AKIAIOSFODNN7EXAMPLE"
        assert (
            config.clip_storage_config.secret_key
            == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        assert config.clip_storage_config.bucket == "my-face-clips"
        assert config.clip_storage_config.url_expiration_s == 3600

    # Test 4: from_settings with optional url_expiration_s
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db4.lance")

        settings = {
            "face_detector": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "face_embedder": {
                "hardware": "N2X/ORCA1",
                "inference_host_address": "@cloud",
            },
            "database": {"db_path": db_path},
            "storage": {
                "endpoint": "./local_storage",
                "access_key": "local_key",
                "secret_key": "local_secret",
                "bucket": "local_clips",
                "url_expiration_s": 3600,
            },
        }

        config = FaceClipManagerConfig.from_settings(settings)

        # Verify storage configuration with expiration
        assert config.clip_storage_config.endpoint == "./local_storage"
        assert config.clip_storage_config.bucket == "local_clips"
        assert config.clip_storage_config.url_expiration_s == 3600

    # Test 5: from_yaml method
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db5.lance").replace("\\", "/")

        yaml_str = f"""
face_detector:
  hardware: N2X/ORCA1
  inference_host_address: "@cloud"
face_embedder:
  hardware: N2X/ORCA1
  inference_host_address: "@cloud"
database:
  db_path: "{db_path}"
  similarity_threshold: 0.7
storage:
  endpoint: "https://minio.example.com"
  access_key: "yaml_access_key"
  secret_key: "yaml_secret_key"
  bucket: "yaml_clips"
  url_expiration_s: 7200
        """

        config, loaded_settings = FaceClipManagerConfig.from_yaml(yaml_str=yaml_str)

        # Verify basic functionality inherited from FaceRecognitionConfig
        assert config.db._threshold == 0.7

        # Verify storage configuration from YAML
        assert config.clip_storage_config.endpoint == "https://minio.example.com"
        assert config.clip_storage_config.access_key == "yaml_access_key"
        assert config.clip_storage_config.secret_key == "yaml_secret_key"
        assert config.clip_storage_config.bucket == "yaml_clips"
        assert config.clip_storage_config.url_expiration_s == 7200
        assert isinstance(loaded_settings, dict)

    # Test 6: Error cases

    # Missing storage configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db6.lance")

        invalid_settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path},
            # Missing storage configuration
        }

        with pytest.raises(Exception):
            FaceClipManagerConfig.from_settings(invalid_settings)

    # Incomplete storage configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db7.lance")

        invalid_settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path},
            "storage": {
                "endpoint": "s3.amazonaws.com",
                "access_key": "test_key",
                # Missing secret_key and bucket
            },
        }

        with pytest.raises(Exception):
            FaceClipManagerConfig.from_settings(invalid_settings)


def test_config_face_tracking():
    """Test FaceTrackingConfig class functionality"""

    # Test 1: Basic instantiation with required parameters
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db.lance")

        # Create model specs
        face_detection_model = degirum_tools.ModelSpec(
            model_name="test_face_detector",
            inference_host_address="@cloud",
            zoo_url="@cloud",
        )
        face_embedding_model = degirum_tools.ModelSpec(
            model_name="test_face_embedder",
            inference_host_address="@cloud",
            zoo_url="@cloud",
        )

        # Create database
        db = ReID_Database(db_path=db_path, threshold=0.5)

        # Create storage config
        clip_storage_config = degirum_tools.ObjectStorageConfig(
            endpoint="http://localhost:9000",
            access_key="testkey",
            secret_key="testsecret",
            bucket="test_bucket",
        )

        # Test basic instantiation
        config = FaceTrackingConfig(
            face_detection_model=face_detection_model,
            face_embedding_model=face_embedding_model,
            db=db,
            clip_storage_config=clip_storage_config,
            credence_count=5,
            alert_mode=AlertMode.ON_UNKNOWNS,
            alert_once=False,
            clip_duration=150,
            video_source=1,
            live_stream_mode="WEB",
            live_stream_rtsp_url="custom_stream",
        )

        # Verify inherited attributes
        assert config.face_detection_model == face_detection_model
        assert config.face_embedding_model == face_embedding_model
        assert config.db == db
        assert config.clip_storage_config == clip_storage_config

        # Verify FaceTrackingConfig specific attributes
        assert config.credence_count == 5
        assert config.alert_mode == AlertMode.ON_UNKNOWNS
        assert config.alert_once is False
        assert config.clip_duration == 150
        assert config.video_source == 1
        assert config.live_stream_mode == "WEB"
        assert config.live_stream_rtsp_url == "custom_stream"

    # Test 2: Default values
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db2.lance")
        db = ReID_Database(db_path=db_path)

        config = FaceTrackingConfig(
            face_detection_model=face_detection_model,
            face_embedding_model=face_embedding_model,
            db=db,
        )

        # Verify default values
        assert config.credence_count == 4
        assert config.alert_mode == AlertMode.NONE
        assert config.alert_once
        assert config.clip_duration == 100
        assert config.notification_config == degirum_tools.notification_config_console
        assert (
            config.notification_message
            == "{time}: Unknown person detected. Saved video: [{filename}]({url})"
        )
        assert config.video_source == 0
        assert config.live_stream_mode == "LOCAL"
        assert config.live_stream_rtsp_url == "face_tracking"

    # Test 3: from_settings method with complete configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db3.lance")

        settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path, "similarity_threshold": 0.6},
            "storage": {
                "endpoint": "s3.amazonaws.com",
                "access_key": "AKIAIOSFODNN7EXAMPLE",
                "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "bucket": "my-face-clips",
                "url_expiration_s": 3600,
            },
            "video_source": "rtsp://example.com/stream",
            "live_stream": {"mode": "WEB", "rtsp_url": "my_custom_stream"},
            "alerts": {
                "credence_count": 8,
                "alert_mode": "ON_ALL",
                "alert_once": False,
                "clip_duration": 200,
                "notification_config": "custom://notification",
                "notification_message": "Custom alert: {time} - {filename}",
            },
        }

        config = FaceTrackingConfig.from_settings(settings)

        # Verify inherited functionality works
        assert config.face_detection_model.model_name == "test_detector"
        assert config.face_embedding_model.model_name == "test_embedder"
        assert config.db._threshold == 0.6
        assert config.clip_storage_config.endpoint == "s3.amazonaws.com"

        # Verify FaceTrackingConfig specific configuration
        assert config.video_source == "rtsp://example.com/stream"
        assert config.live_stream_mode == "WEB"
        assert config.live_stream_rtsp_url == "my_custom_stream"
        assert config.credence_count == 8
        assert config.alert_mode == AlertMode.ON_ALL
        assert config.alert_once is False
        assert config.clip_duration == 200
        assert config.notification_config == "custom://notification"
        assert config.notification_message == "Custom alert: {time} - {filename}"

    # Test 4: from_settings with different alert modes
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db4.lance")

        alert_modes = ["ON_UNKNOWNS", "ON_KNOWNS", "ON_ALL", "NONE"]

        for alert_mode_str in alert_modes:
            settings = {
                "face_detector": {
                    "hardware": "N2X/ORCA1",
                    "inference_host_address": "@cloud",
                },
                "face_embedder": {
                    "hardware": "N2X/ORCA1",
                    "inference_host_address": "@cloud",
                },
                "database": {"db_path": db_path},
                "storage": {
                    "endpoint": "./local_storage",
                    "access_key": "local_key",
                    "secret_key": "local_secret",
                    "bucket": "local_clips",
                    "url_expiration_s": 3600,
                },
                "video_source": 0,
                "live_stream": {"mode": "LOCAL"},
                "alerts": {"alert_mode": alert_mode_str},
            }

            config = FaceTrackingConfig.from_settings(settings)
            assert config.alert_mode == AlertMode[alert_mode_str]

    # Test 5: from_yaml method
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db5.lance").replace("\\", "/")

        yaml_str = f"""
face_detector:
  hardware: N2X/ORCA1
  inference_host_address: "@cloud"
face_embedder:
  hardware: N2X/ORCA1
  inference_host_address: "@cloud"
database:
  db_path: "{db_path}"
  similarity_threshold: 0.7
storage:
  endpoint: "https://minio.example.com"
  access_key: "yaml_access_key"
  secret_key: "yaml_secret_key"
  bucket: "yaml_clips"
  url_expiration_s: 7200
video_source: "/path/to/video.mp4"
live_stream:
  mode: "NONE"
  rtsp_url: "yaml_stream"
alerts:
  credence_count: 6
  alert_mode: "ON_KNOWNS"
  alert_once: true
  clip_duration: 120
  notification_config: "yaml://notification"
  notification_message: "YAML alert: {{time}}"
"""

        config, loaded_settings = FaceTrackingConfig.from_yaml(yaml_str=yaml_str)

        # Verify basic functionality inherited from parent classes
        assert config.db._threshold == 0.7
        assert config.clip_storage_config.bucket == "yaml_clips"

        # Verify FaceTrackingConfig configuration from YAML
        assert config.video_source == "/path/to/video.mp4"
        assert config.live_stream_mode == "NONE"
        assert config.live_stream_rtsp_url == "yaml_stream"
        assert config.credence_count == 6
        assert config.alert_mode == AlertMode.ON_KNOWNS
        assert config.alert_once
        assert config.clip_duration == 120
        assert config.notification_config == "yaml://notification"
        assert config.notification_message == "YAML alert: {time}"
        assert isinstance(loaded_settings, dict)

    # Test 6: Error cases

    # Missing required video_source
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db6.lance")

        invalid_settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path},
            "storage": {
                "endpoint": "s3.amazonaws.com",
                "access_key": "test_key",
                "secret_key": "test_secret",
                "bucket": "test_bucket",
                "url_expiration_s": 3600,
            },
            "live_stream": {"mode": "LOCAL"},
            "alerts": {"alert_mode": "ON_UNKNOWNS"},
            # Missing video_source
        }

        with pytest.raises(Exception):
            FaceTrackingConfig.from_settings(invalid_settings)

    # Missing required live_stream
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db7.lance")

        invalid_settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path},
            "storage": {
                "endpoint": "s3.amazonaws.com",
                "access_key": "test_key",
                "secret_key": "test_secret",
                "bucket": "test_bucket",
                "url_expiration_s": 3600,
            },
            "video_source": 0,
            "alerts": {"alert_mode": "ON_UNKNOWNS"},
            # Missing live_stream
        }

        with pytest.raises(Exception):
            FaceTrackingConfig.from_settings(invalid_settings)

    # Missing required alerts
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db8.lance")

        invalid_settings = {
            "face_detector": {
                "model_name": "test_detector",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "face_embedder": {
                "model_name": "test_embedder",
                "inference_host_address": "@cloud",
                "model_zoo_url": "@cloud",
            },
            "database": {"db_path": db_path},
            "storage": {
                "endpoint": "s3.amazonaws.com",
                "access_key": "test_key",
                "secret_key": "test_secret",
                "bucket": "test_bucket",
                "url_expiration_s": 3600,
            },
            "video_source": 0,
            "live_stream": {"mode": "LOCAL"},
            # Missing alerts
        }

        with pytest.raises(Exception):
            FaceTrackingConfig.from_settings(invalid_settings)
