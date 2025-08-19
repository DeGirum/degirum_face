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


class FaceDetector:
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

    @classmethod
    def _load_auto_mode_model(
        cls, hardware: str, inference_host_address: str = "@cloud"
    ) -> Tuple[Any, str, str]:
        """
        Common logic for auto mode model loading.

        Args:
            hardware: AI hardware device
            inference_host_address: Where to run inference

        Returns:
            Tuple of (model, model_name, zoo_url)
        """
        config = get_model_config()

        # Validate hardware supports the task
        if not config.validate_hardware_task_combination(hardware, cls.TASK):
            available_tasks = config.get_tasks_for_hardware(hardware)
            raise ValueError(
                f"Hardware '{hardware}' does not support task '{cls.TASK}'. "
                f"Available tasks: {available_tasks}"
            )

        # Get zoo URL and model
        model_name = config.get_default_model(hardware, cls.TASK)

        if not model_name:
            # Get first available model if no default
            available_models = config.get_models_for_task_and_hardware(
                cls.TASK, hardware
            )
            if available_models:
                model_name = available_models[0]
                logger.warning(
                    f"No default model for {hardware}/{cls.TASK}, using {model_name}"
                )
            else:
                raise ValueError(
                    f"No models available for {hardware}/{cls.TASK}. "
                    f"Available hardware: {config.get_hardware_for_task(cls.TASK)}"
                )

        # Get the zoo_url for this specific model
        zoo_url = config.get_model_zoo_url(model_name)

        # Load the model
        try:
            zoo = dg.connect(inference_host_address, zoo_url)
            model = zoo.load_model(model_name)
            logger.info(
                f"Auto mode: Loaded '{model_name}' on {hardware} via {inference_host_address}"
            )
            return model, model_name, zoo_url

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}' for {hardware}: {e}"
            ) from e

    def __init__(
        self,
        hardware: Optional[str] = None,
        *,
        model=None,
        model_name=None,
        zoo_url=None,
        inference_host_address="@cloud",
        _creation_mode="auto",  # Internal parameter to track creation mode
    ):
        """
        Create a face detector. Defaults to auto mode if only hardware is provided.

        Args:
            hardware: AI hardware device for auto mode (e.g., "hailo8", "degirum_orca")
            model: Pre-loaded PySDK model (for internal use by factory methods)
            model_name: Model name (internal use)
            zoo_url: Zoo URL (internal use)
            inference_host_address: Where to run inference (default: "@cloud")

        Examples:
            # Auto mode (default) - most convenient
            detector = FaceDetector(hardware="hailo8")
            detector = FaceDetector("degirum_orca")  # positional also works

            # Or use explicit factory methods
            detector = FaceDetector.custom(model_name="...", zoo_url="...")
            detector = FaceDetector.from_config(hardware="hailo8", model_name="...")
        """
        # If model is provided, this is from a factory method
        if model is not None:
            # This path is for factory methods - model is already loaded
            self.model = model
            self.hardware = hardware
            self.model_name = model_name
            self.zoo_url = zoo_url
            self.inference_host_address = inference_host_address
            self._creation_mode = _creation_mode
            return

        # Otherwise, this is direct instantiation - use auto mode
        if hardware is None:
            raise ValueError(
                "Hardware must be specified for auto mode. Use:\n"
                "  FaceDetector(hardware='hailo8') or\n"
                "  FaceDetector.custom(model_name='...', zoo_url='...') or\n"
                "  FaceDetector.from_config(hardware='...', model_name='...')"
            )

        # Auto mode initialization - this is the default behavior for direct instantiation
        config = get_model_config()

        # Validate hardware supports the task
        if not config.validate_hardware_task_combination(hardware, self.TASK):
            available_tasks = config.get_tasks_for_hardware(hardware)
            raise ValueError(
                f"Hardware '{hardware}' does not support task '{self.TASK}'. "
                f"Available tasks: {available_tasks}"
            )

        # Get zoo URL and model
        model_name = config.get_default_model(hardware, self.TASK)

        if not model_name:
            # Get first available model if no default
            available_models = config.get_models_for_task_and_hardware(
                self.TASK, hardware
            )
            if available_models:
                model_name = available_models[0]
                logger.warning(
                    f"No default model for {hardware}/{self.TASK}, using {model_name}"
                )
            else:
                raise ValueError(
                    f"No models available for {hardware}/{self.TASK}. "
                    f"Available hardware: {config.get_hardware_for_task(self.TASK)}"
                )

        # Get the zoo_url for this specific model
        zoo_url = config.get_model_zoo_url(model_name)

        # Load the model
        try:
            zoo = dg.connect(inference_host_address, zoo_url)
            model = zoo.load_model(model_name)
            logger.info(
                f"Auto mode: Loaded '{model_name}' on {hardware} via {inference_host_address}"
            )

            # Set instance variables
            self.model = model
            self.hardware = hardware
            self.model_name = model_name
            self.zoo_url = zoo_url
            self.inference_host_address = inference_host_address
            self._creation_mode = "auto"  # Direct constructor call is auto mode

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}' for {hardware}: {e}"
            ) from e

    @classmethod
    def auto(
        cls,
        hardware: str,
        inference_host_address: str = "@cloud",
    ):
        """
        Create a face detector using automatic model selection for the specified hardware.

        This is an explicit factory method for auto mode - same as calling FaceDetector(hardware="...").

        Args:
            hardware: AI hardware device (e.g., "hailo8", "degirum_orca", "cpu")
            inference_host_address: Where to run inference (default: "@cloud")

        Returns:
            FaceDetector instance with automatically selected model

        Example:
            detector = FaceDetector.auto(hardware="hailo8")
            # This is equivalent to: detector = FaceDetector(hardware="hailo8")
        """
        # Use the shared helper method to avoid duplication
        model, model_name, zoo_url = cls._load_auto_mode_model(
            hardware, inference_host_address
        )

        return cls(
            hardware=hardware,
            model=model,
            model_name=model_name,
            zoo_url=zoo_url,
            inference_host_address=inference_host_address,
            _creation_mode="auto_factory",
        )

    @classmethod
    def custom(cls, model_spec: ModelSpec):
        """
        Create a face detector with a custom model specification.

        Args:
            model_spec: ModelSpec with complete model configuration

        Returns:
            FaceDetector instance with the specified custom model

        Example:
            spec = ModelSpec(
                model_name="yolo_v8n_face_detection--512x512_quant_n2x_orca1_bgr",
                zoo_url="https://cs.degirum.com/degirum/orca",
                inference_host_address="@localhost"
            )
            detector = FaceDetector.custom(spec)
        """
        try:
            # ModelSpec handles both connection and loading
            model = model_spec.load_model()
            logger.info(
                f"Custom mode: Loaded '{model_spec.model_name}' from {model_spec.zoo_url} via {model_spec.inference_host_address}"
            )

            return cls(
                hardware=None,
                model=model,
                model_name=model_spec.model_name,
                zoo_url=model_spec.zoo_url,
                inference_host_address=model_spec.inference_host_address,
                _creation_mode="custom",
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load custom model '{model_spec.model_name}' from {model_spec.zoo_url}: {e}"
            ) from e

    @classmethod
    def from_config(
        cls,
        model_name: str,
        inference_host_address: str = "@cloud",
    ):
        """
        Create a face detector using a specific model from the built-in configuration.
        All model information (hardware, zoo_url, task) is automatically extracted from YAML.

        Args:
            model_name: Specific model name from the configuration
            inference_host_address: Where to run inference (default: "@cloud")

        Returns:
            FaceDetector instance with the specified model from config

        Example:
            detector = FaceDetector.from_config(
                model_name="yolo_v8n_face_detection--512x512_quant_n2x_hailo8_bgr"
            )
        """
        config = get_model_config()

        # Get model info from YAML - this contains everything we need
        model_info = config.get_model_info(model_name)
        if not model_info:
            available_models = config.get_models_for_task(cls.TASK)
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available face detection models: {available_models}"
            )

        # Extract all info from YAML
        hardware = model_info.get("hardware")
        task = model_info.get("task")
        zoo_url = model_info.get("zoo_url", "degirum/public")

        # Simple validation - just check if this is a face detection model
        if task != cls.TASK:
            raise ValueError(
                f"Model '{model_name}' is for task '{task}', not '{cls.TASK}'. "
                f"Use FaceDetector only for face detection models."
            )

        # Load the model using info from YAML
        try:
            zoo = dg.connect(inference_host_address, zoo_url)
            model = zoo.load_model(model_name)
            logger.info(
                f"Config mode: Loaded '{model_name}' (hardware: {hardware}) via {inference_host_address}"
            )

            return cls(
                hardware=hardware,
                model=model,
                model_name=model_name,
                zoo_url=zoo_url,
                inference_host_address=inference_host_address,
                _creation_mode="from_config",
            )

        except ConnectionError as e:
            raise RuntimeError(f"Failed to connect to zoo at '{zoo_url}': {e}") from e
        except Exception as e:
            # Check if it's a model loading error vs connection error
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                raise RuntimeError(
                    f"Model '{model_name}' not found in zoo '{zoo_url}': {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' from {zoo_url}: {e}"
                ) from e

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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "hardware": self.hardware,
            "inference_host": self.inference_host_address,
            "zoo_url": self.zoo_url,
            "task": self.TASK,
            "creation_mode": self._creation_mode,
        }

    @staticmethod
    def get_supported_hardware() -> List[str]:
        """Get list of hardware devices that support face detection."""
        config = get_model_config()
        return config.get_hardware_for_task(FaceDetector.TASK)

    @staticmethod
    def get_available_models(
        hardware: Optional[str] = None,
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get list of available models for face detection.

        Args:
            hardware: Optional hardware device. If provided, returns models for that hardware.
                     If None, returns all models for all hardware.

        Returns:
            If hardware is specified: List of model names for that hardware
            If hardware is None: Dict mapping hardware -> list of model names

        Examples:
            # Get models for specific hardware
            models = FaceDetector.get_available_models("hailo8")

            # Get all models for all hardware
            all_models = FaceDetector.get_available_models()
            # Returns: {"hailo8": [...], "degirum_orca": [...], ...}
        """
        config = get_model_config()

        if hardware is not None:
            return config.get_models_for_task_and_hardware(FaceDetector.TASK, hardware)
        else:
            # Return all models for all hardware
            all_hardware = config.get_hardware_for_task(FaceDetector.TASK)
            result = {}
            for hw in all_hardware:
                models = config.get_models_for_task_and_hardware(FaceDetector.TASK, hw)
                if models:  # Only include hardware that has models
                    result[hw] = models
            return result

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
