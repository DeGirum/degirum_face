#
# face_detector.py: Clean face detection API with auto mode and factory methods
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


class FaceDetector(BaseModelComponent):

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
    DeGirum Face Detection API with automatic model loading and PySDK integration.

    The FaceDetector provides three ways to create a detector:

    1. Auto Mode (Default): Automatically selects the best available model for your hardware
        ```python
        detector = FaceDetector("hailo8")           # Auto mode - recommended & default
        detector = FaceDetector(hardware="hailo8")  # Same as above, explicit
        ```

    2. Factory Methods: Explicit creation patterns for clarity
        ```python
        detector = FaceDetector.auto("hailo8")      # Explicit auto mode (same as #1)
        detector = FaceDetector.custom(model_spec)  # Custom model specification
        detector = FaceDetector.from_config(model_name)  # From built-in config
        ```

    3. All creation methods support the same detection interface:
        ```python
        results = detector.detect("image.jpg")           # Single prediction
        batch_results = detector.detect_batch(images)    # Batch prediction
        stream_results = detector.predict_stream(video)  # Stream prediction
        ```

    The detector automatically handles:
    - Model loading and caching for efficient reuse
    - Image input validation (paths, URLs, numpy arrays)
    - Hardware compatibility checking
    - PySDK native inference results with overlay methods

    Returns PySDK inference results that include image_overlay() methods and detection data.
    """

    # Class constant - this is a face detection class
    TASK = "face_detection"

    # All factory methods and model info methods are now inherited from BaseModelComponent

    def detect(self, image: Union[str, Path, np.ndarray]):
        """
        Detect faces in a single image.

        Args:
            image: Path to image file, URL to image, or numpy array

        Returns:
            PySDK inference result with detection data and overlay methods
        """
        return self.model.predict(image)

    def detect_batch(self, images: List[Union[str, Path, np.ndarray]]) -> List:
        """
        Detect faces in multiple images efficiently.

        Args:
            images: List of image paths, URLs, or numpy arrays

        Returns:
            List of PySDK inference results (one per image)

        Raises:
            ValueError: If images is not a list or is empty
        """
        if not isinstance(images, list):
            raise ValueError(f"Expected list of images, got {type(images)}")

        if not images:
            logger.warning("Empty image list provided to detect_batch")
            return []

        results = []
        for i, image in enumerate(images):
            try:
                result = self.detect(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                # You might want to decide whether to continue or raise here
                # For now, we'll continue and let the caller handle partial results
                results.append(None)

        return results

    def predict_stream(
        self, image_generator: Iterator[Union[str, Path, np.ndarray]]
    ) -> Generator[Any, None, None]:
        """
        Process a stream of images (generator) for real-time applications.

        Args:
            image_generator: Generator yielding images

        Yields:
            PySDK inference results for each image

        Raises:
            ValueError: If image_generator is not iterable
        """
        try:
            # Test if it's iterable
            iter(image_generator)
        except TypeError:
            raise ValueError("image_generator must be iterable")

        for image in image_generator:
            try:
                yield self.detect(image)
            except Exception as e:
                logger.error(f"Failed to process image in stream: {e}")
                # Yield None or skip - depending on your requirements
                # For now, we'll skip failed images
                continue

    def __repr__(self) -> str:
        return f"FaceDetector[{self._creation_mode}](model='{self.model_name}')"


# Convenience functions for backward compatibility and simple use cases
def detect_faces(
    image: Union[str, Path, np.ndarray],
    hardware: str = "degirum_orca",
    inference_host_address: str = "@cloud",
):
    """
    Convenience function to detect faces in a single image using auto mode.

    Note: For multiple predictions, use FaceDetector class for better performance.

    Args:
        image: Path to image file, URL to image, or numpy array
        hardware: AI hardware device to use (default: "degirum_orca")
        inference_host_address: Where to run inference (default: "@cloud")

    Returns:
        PySDK inference result with detection data and overlay methods

    Example:
        result = detect_faces("image.jpg", hardware="hailo8")
    """
    detector = FaceDetector.auto(
        hardware=hardware, inference_host_address=inference_host_address
    )
    return detector.detect(image)
