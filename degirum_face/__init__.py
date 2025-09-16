#
# Face tracking application package
# Copyright DeGirum Corp. 2025
#
# Implements various classes and functions for face tracking application development
#


def __dir__():
    return [
        "model_registry",
        "get_face_detection_model_spec",
        "get_face_embedding_model_spec",
        "AlertMode",
        "FaceRecognitionConfig",
        "FaceClipManagerConfig",
        "FaceTrackingConfig",
        "FaceFilterConfig",
        "FaceRecognition",
        "FaceRecognitionResult",
        "FaceClipManager",
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


from .reid_database import ReID_Database  # noqa

from .face_tracking_gizmos import (  # noqa
    ObjectMap,
    AlertMode,
    FaceSearchGizmo,
    FaceExtractGizmo,
    ObjectAnnotateGizmo,
)

from .configs import (  # noqa
    model_registry,
    get_face_detection_model_spec,
    get_face_embedding_model_spec,
    FaceRecognitionConfig,
    FaceClipManagerConfig,
    FaceTrackingConfig,
    FaceFilterConfig,
)

from .face_tracking import (  # noqa
    FaceRecognition,
    FaceClipManager,
    start_face_tracking_pipeline,
)

from .face_data import FaceRecognitionResult  # noqa

from .logging_config import (  # noqa
    configure_logging,
    set_log_level,
    logging_disable,
)
