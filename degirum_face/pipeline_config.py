"""
Pipeline model configuration for FaceRecognition system.

This module defines model specification structures that allow
complete freedom for each model to have its own zoo_url and inference_host_address.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import degirum as dg


@dataclass
class ModelSpec:
    """
    Complete specification for a single model with extensible parameters.

    This allows complete freedom - each model can have its own
    zoo_url, inference_host_address, and additional parameters for connection and loading.

    Attributes:
        model_name: Exact model name to load
        zoo_url: Zoo URL where this model is hosted
        inference_host_address: Where to run inference for this model
        connect_kwargs: Additional keyword arguments for zoo connection (e.g., token, etc.)
        load_kwargs: Additional keyword arguments for model loading

    Example:
        >>> # Basic usage
        >>> detector_spec = ModelSpec(
        ...     model_name="yolo_v8n_face_detection--512x512_quant_n2x_orca1_bgr",
        ...     zoo_url="https://cs.degirum.com/degirum/orca",
        ...     inference_host_address="@localhost"
        ... )

        >>> # With token for connection
        >>> private_spec = ModelSpec(
        ...     model_name="private_face_detector_v2",
        ...     zoo_url="https://private.company.com/models",
        ...     inference_host_address="@cloud",
        ...     connect_kwargs={"token": "secret_token"}
        ... )

        >>> # With both connection and loading parameters
        >>> advanced_spec = ModelSpec(
        ...     model_name="advanced_model",
        ...     zoo_url="https://api.company.com/models",
        ...     connect_kwargs={"token": "auth_token", "timeout": 30},
        ...     load_kwargs={"version": "latest", "precision": "fp16"}
        ... )
    """

    model_name: str
    zoo_url: str = "degirum/public"
    inference_host_address: str = "@cloud"
    connect_kwargs: Dict[str, Any] = field(default_factory=dict)
    load_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model specification after initialization."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.zoo_url:
            raise ValueError("zoo_url cannot be empty")
        if not self.inference_host_address:
            raise ValueError("inference_host_address cannot be empty")
        if self.connect_kwargs is None:
            self.connect_kwargs = {}
        if self.load_kwargs is None:
            self.load_kwargs = {}

    def load_model(self, zoo=None):
        """
        Load the model using this specification.

        Args:
            zoo: Optional pre-connected zoo instance. If None, will connect automatically.

        Returns:
            Loaded model instance
        """
        if zoo is None:
            # Connect with connect_kwargs (this is where token should go)
            zoo = dg.connect(
                self.inference_host_address, self.zoo_url, **self.connect_kwargs
            )

        # Load model with load_kwargs
        return zoo.load_model(self.model_name, **self.load_kwargs)

    def load_model_direct(self):
        """
        Alternative: Load model directly using PySDK load_model API without separate connect/load stages.

        This combines both connect_kwargs and load_kwargs into a single call.

        Returns:
            Loaded model instance
        """
        # Combine all kwargs for direct loading
        all_kwargs = {**self.connect_kwargs, **self.load_kwargs}
        return dg.load_model(
            self.model_name,
            zoo_url=self.zoo_url,
            inference_host_address=self.inference_host_address,
            **all_kwargs
        )


@dataclass
class PipelineModelConfig:
    """
    Configuration for face recognition pipeline using ModelSpec for each component.

    This allows complete flexibility - detector and embedder can use
    different zoo_urls and inference_host_addresses.

    Attributes:
        detector_spec: Complete specification for face detector model
        embedder_spec: Complete specification for face embedder model

    Example:
        >>> detector_spec = ModelSpec(
        ...     model_name="yolo_v8n_face_detection--512x512_quant_n2x_orca1_bgr",
        ...     zoo_url="https://cs.degirum.com/degirum/orca",
        ...     inference_host_address="@localhost"
        ... )
        >>> embedder_spec = ModelSpec(
        ...     model_name="face_embedding_mobilenet--112x112_quant_n2x_orca1_bgr",
        ...     zoo_url="https://cs.degirum.com/degirum/other_zoo",
        ...     inference_host_address="@cloud"
        ... )
        >>> config = PipelineModelConfig(detector_spec, embedder_spec)
        >>> face_rec = FaceRecognition.custom(config)
    """

    detector_spec: ModelSpec
    embedder_spec: ModelSpec

    def __post_init__(self):
        """Validate pipeline configuration after initialization."""
        if not isinstance(self.detector_spec, ModelSpec):
            raise ValueError("detector_spec must be a ModelSpec instance")
        if not isinstance(self.embedder_spec, ModelSpec):
            raise ValueError("embedder_spec must be a ModelSpec instance")
