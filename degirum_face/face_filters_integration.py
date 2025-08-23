"""
Integration utilities for face filtering in the face recognition pipeline.
"""

from .face_filters import FaceFilter, FaceFilterConfig


# Provide a default FaceFilter instance (can be replaced by user)
def get_default_face_filter():
    return FaceFilter(FaceFilterConfig())
