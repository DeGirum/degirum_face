#
# face_embedder.py: Clean face embedding API with auto mode and factory methods
#
# Copyright DeGirum Corporation 2025
#

import numpy as np
import degirum as dg
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple, Iterator, Generator

from .model_config import get_model_config
from .pipeline_config import ModelSpec

logger = logging.getLogger(__name__)


from .base_model_component import BaseModelComponent


class FaceEmbedder(BaseModelComponent):

    def __init__(
        self,
        hardware: str = None,
        model: Any = None,
        model_name: str = None,
        zoo_url: str = None,
        inference_host_address: str = None,
        _creation_mode: str = None,
    ):
        self.hardware = hardware
        self.model = model
        self.model_name = model_name
        self.zoo_url = zoo_url
        self.inference_host_address = inference_host_address
        self._creation_mode = _creation_mode

    """
    DeGirum Face Embedding API with automatic model loading and PySDK integration.

    The FaceEmbedder provides three ways to create an embedder:

    1. Auto Mode (Default): Automatically selects the best available model for your hardware
        ```python
        embedder = FaceEmbedder("hailo8")           # Auto mode - recommended & default
        embedder = FaceEmbedder(hardware="hailo8")  # Same as above, explicit
        ```

    2. Factory Methods: Explicit creation patterns for clarity
        ```python
        embedder = FaceEmbedder.auto("hailo8")      # Explicit auto mode (same as #1)
        embedder = FaceEmbedder.custom(model_spec)  # Custom model specification
        embedder = FaceEmbedder.from_config(model_name)  # From built-in config
        ```

    3. All creation methods support the same embedding interface:
        ```python
        embeddings = embedder.embed("image.jpg")           # Single embedding
        batch_embeddings = embedder.embed_batch(images)    # Batch embedding
        stream_embeddings = embedder.embed_stream(video)   # Stream embedding
        similarity = embedder.compare_embeddings(emb1, emb2)  # Compare embeddings
        ```

    The embedder automatically handles:
    - Model loading and caching for efficient reuse
    - Image input validation (paths, URLs, numpy arrays)
    - Hardware compatibility checking
    - PySDK native inference results with embedding extraction methods
    - Embedding similarity calculations

    Returns PySDK inference results that include embedding vectors and similarity methods.
    """

    # Class constant - this is a face recognition/embedding class
    TASK = "face_embedding"

    # All factory methods and model info methods are now inherited from BaseModelComponent

    def embed(self, image: Union[str, Path, np.ndarray]):
        """
        Generate face embedding for a single image.

        Args:
            image: Path to image file, URL to image, or numpy array

        Returns:
            PySDK inference result with embedding data
        """
        return self.model.predict(image)

    def embed_batch(self, images: List[Union[str, Path, np.ndarray]]) -> List:
        """
        Generate face embeddings for multiple images efficiently.

        Args:
            images: List of image paths, URLs, or numpy arrays

        Returns:
            List of PySDK inference results (one per image)
        """
        return [self.embed(image) for image in images]

    def embed_stream(self, image_generator):
        """
        Process a stream of images (generator) for real-time embedding generation.

        Args:
            image_generator: Generator yielding images

        Yields:
            PySDK inference results for each image
        """
        for image in image_generator:
            yield self.embed(image)

    def __repr__(self) -> str:
        return f"FaceEmbedder[{self._creation_mode}](model='{self.model_name}')"


# Convenience functions for backward compatibility and simple use cases
def embed_face(
    image: Union[str, Path, np.ndarray],
    hardware: str = "degirum_orca",
    inference_host_address: str = "@cloud",
):
    """
    Convenience function to generate face embedding for a single image using auto mode.

    Note: For multiple predictions, use FaceEmbedder class for better performance.

    Args:
        image: Path to image file, URL to image, or numpy array
        hardware: AI hardware device to use (default: "degirum_orca")
        inference_host_address: Where to run inference (default: "@cloud")

    Returns:
        PySDK inference result with embedding data

    Example:
        result = embed_face("image.jpg", hardware="hailo8")
        embedding_vector = embedder.extract_embedding_vector(result)
    """
    embedder = FaceEmbedder.auto(
        hardware=hardware, inference_host_address=inference_host_address
    )
    return embedder.embed(image)


def verify_faces(
    image1: Union[str, Path, np.ndarray],
    image2: Union[str, Path, np.ndarray],
    hardware: str = "degirum_orca",
    threshold: float = 0.5,
    inference_host_address: str = "@cloud",
) -> Dict[str, Any]:
    """
    Convenience function to verify if two images contain the same person.

    Args:
        image1: First image (path, URL, or numpy array)
        image2: Second image (path, URL, or numpy array)
        hardware: AI hardware device to use (default: "degirum_orca")
        threshold: Similarity threshold for positive verification
        inference_host_address: Where to run inference (default: "@cloud")

    Returns:
        Dictionary with verification result and similarity score

    Example:
        result = verify_faces("person1.jpg", "person2.jpg", hardware="hailo8")
        print(f"Same person: {result['is_same_person']}")
        print(f"Similarity: {result['similarity_score']:.3f}")
    """
    embedder = FaceEmbedder.auto(
        hardware=hardware, inference_host_address=inference_host_address
    )
    return embedder.verify_faces(image1, image2, threshold=threshold)
