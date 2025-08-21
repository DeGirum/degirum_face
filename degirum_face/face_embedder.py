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


class FaceEmbedder:
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
        Create a face embedder. Defaults to auto mode if only hardware is provided.

        Args:
            hardware: AI hardware device for auto mode (e.g., "hailo8", "degirum_orca")
            model: Pre-loaded PySDK model (for internal use by factory methods)
            model_name: Model name (internal use)
            zoo_url: Zoo URL (internal use)
            inference_host_address: Where to run inference (default: "@cloud")

        Examples:
            # Auto mode (default) - most convenient
            embedder = FaceEmbedder(hardware="hailo8")
            embedder = FaceEmbedder("degirum_orca")  # positional also works

            # Or use explicit factory methods
            embedder = FaceEmbedder.custom(model_name="...", zoo_url="...")
            embedder = FaceEmbedder.from_config(model_name="...")
        """
        # If model is provided, this is from a factory method
        if model is not None:
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
                "  FaceEmbedder(hardware='hailo8') or\n"
                "  FaceEmbedder.custom(model_name='...', zoo_url='...') or\n"
                "  FaceEmbedder.from_config(hardware='...', model_name='...')"
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
        Create a face embedder using automatic model selection for the specified hardware.

        This is an explicit factory method for auto mode - same as calling FaceEmbedder(hardware="...").

        Args:
            hardware: AI hardware device (e.g., "hailo8", "degirum_orca", "cpu")
            inference_host_address: Where to run inference (default: "@cloud")

        Returns:
            FaceEmbedder instance with automatically selected model

        Example:
            embedder = FaceEmbedder.auto(hardware="hailo8")
            # This is equivalent to: embedder = FaceEmbedder(hardware="hailo8")
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
        Create a face embedder with a custom model specification.

        Args:
            model_spec: ModelSpec with complete model configuration

        Returns:
            FaceEmbedder instance with the specified custom model

        Example:
            spec = ModelSpec(
                model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8_1",
                zoo_url="degirum/hailo",
                inference_host_address="@localhost"
            )
            embedder = FaceEmbedder.custom(spec)
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
        Create a face embedder using a specific model from the built-in configuration.
        All model information (hardware, zoo_url, task) is automatically extracted from YAML.

        Args:
            model_name: Specific model name from the configuration
            inference_host_address: Where to run inference (default: "@cloud")

        Returns:
            FaceEmbedder instance with the specified model from config

        Example:
            embedder = FaceEmbedder.from_config(
                model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8_1"
            )
        """
        config = get_model_config()

        # Get model info from YAML - this contains everything we need
        model_info = config.get_model_info(model_name)
        if not model_info:
            available_models = config.get_models_for_task(cls.TASK)
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available face recognition models: {available_models}"
            )

        # Extract all info from YAML
        hardware = model_info.get("hardware")
        task = model_info.get("task")
        zoo_url = model_info.get("zoo_url", "degirum/public")

        # Simple validation - just check if this is a face recognition model
        if task != cls.TASK:
            raise ValueError(
                f"Model '{model_name}' is for task '{task}', not '{cls.TASK}'. "
                f"Use FaceEmbedder only for face recognition models."
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

    def extract_embedding_vector(self, result) -> np.ndarray:
        """
        Extract the embedding vector from a PySDK inference result.

        Args:
            result: PySDK inference result from embed() method

        Returns:
            Numpy array containing the face embedding vector
        """
        # Extract embedding from PySDK result structure: result.results[0]["data"]
        try:
            # Primary method: PySDK face embedding results structure
            if hasattr(result, "results") and len(result.results) > 0:
                embedding = result.results[0]["data"]
                # Handle both numpy arrays and list formats
                embedding = (
                    embedding.squeeze()
                    if isinstance(embedding, np.ndarray)
                    else embedding[0]
                )
                return np.array(embedding)

            # Fallback methods for different result structures
            elif hasattr(result, "embedding"):
                return np.array(result.embedding)
            elif hasattr(result, "output"):
                # If output is a list/array, take the first element
                output = result.output
                if isinstance(output, list) and len(output) > 0:
                    return np.array(output[0])
                return np.array(output)
            else:
                # Last resort: try to convert the result directly
                return np.array(result)

        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(
                f"Unable to extract embedding vector from result. "
                f"Expected result.results[0]['data'] structure, got: {type(result)}. "
                f"Error: {e}"
            )

    def compare_embeddings(
        self,
        embedding1: Union[np.ndarray, Any],
        embedding2: Union[np.ndarray, Any],
        metric: str = "cosine",
    ) -> float:
        """
        Compare two face embeddings and return similarity score.

        Args:
            embedding1: First embedding (numpy array or PySDK result)
            embedding2: Second embedding (numpy array or PySDK result)
            metric: Similarity metric ("cosine", "euclidean", "dot_product")

        Returns:
            Similarity score (higher values indicate more similar faces)
        """
        # Extract vectors if they're PySDK results
        if not isinstance(embedding1, np.ndarray):
            embedding1 = self.extract_embedding_vector(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = self.extract_embedding_vector(embedding2)

        # Normalize embeddings for better comparison
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        if metric == "cosine":
            # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
            return np.dot(embedding1, embedding2)
        elif metric == "euclidean":
            # Euclidean distance (convert to similarity: lower distance = higher similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)  # Convert distance to similarity
        elif metric == "dot_product":
            # Dot product similarity
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(
                f"Unsupported metric: {metric}. Use 'cosine', 'euclidean', or 'dot_product'"
            )

    def verify_faces(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        threshold: float = 0.5,
        metric: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Verify if two images contain the same person.

        Args:
            image1: First image (path, URL, or numpy array)
            image2: Second image (path, URL, or numpy array)
            threshold: Similarity threshold for positive verification
            metric: Similarity metric to use

        Returns:
            Dictionary with verification result, similarity score, and confidence
        """
        # Generate embeddings for both images
        result1 = self.embed(image1)
        result2 = self.embed(image2)

        # Compare embeddings
        similarity = self.compare_embeddings(result1, result2, metric=metric)

        # Make verification decision
        is_same_person = similarity >= threshold

        return {
            "is_same_person": is_same_person,
            "similarity_score": float(similarity),
            "threshold": threshold,
            "metric": metric,
            "confidence": float(abs(similarity - threshold)),  # Distance from threshold
        }

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
        """Get list of hardware devices that support face embedding/recognition."""
        config = get_model_config()
        return config.get_hardware_for_task(FaceEmbedder.TASK)

    @staticmethod
    def get_available_models(
        hardware: Optional[str] = None,
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get list of available models for face embedding/recognition.

        Args:
            hardware: Optional hardware device. If provided, returns models for that hardware.
                     If None, returns all models for all hardware.

        Returns:
            If hardware is specified: List of model names for that hardware
            If hardware is None: Dict mapping hardware -> list of model names

        Examples:
            # Get models for specific hardware
            models = FaceEmbedder.get_available_models("hailo8")

            # Get all models for all hardware
            all_models = FaceEmbedder.get_available_models()
            # Returns: {"hailo8": [...], "degirum_orca": [...], ...}
        """
        config = get_model_config()

        if hardware is not None:
            return config.get_models_for_task_and_hardware(FaceEmbedder.TASK, hardware)
        else:
            # Return all models for all hardware
            all_hardware = config.get_hardware_for_task(FaceEmbedder.TASK)
            result = {}
            for hw in all_hardware:
                models = config.get_models_for_task_and_hardware(FaceEmbedder.TASK, hw)
                if models:  # Only include hardware that has models
                    result[hw] = models
            return result

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
