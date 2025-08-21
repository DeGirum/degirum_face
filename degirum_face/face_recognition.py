#
# face_recognition.py: Comprehensive face recognition pipeline
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# This module provides a complete face recognition pipeline that integrates
# face detection, embedding generation, and database management into a
# unified API for face enrollment, verification, and identification.
#
# Key Features:
# - Face Enrollment: Register new faces with quality validation
# - Face Verification: 1:1 verification with confidence scoring
# - Face Identification: 1:N identification with ranking
# - Quality Control: Automatic face quality assessment and filtering
# - Database Management: Persistent storage with deduplication
# - Auto-configuration: Hardware-optimized model selection
#
# Usage Examples:
#     from degirum_face import FaceRecognition, PipelineModelConfig
#
#     # Auto mode with cloud inference (recommended)
#     face_rec = FaceRecognition.auto("hailo8")
#
#     # Auto mode with local inference
#     face_rec = FaceRecognition.auto("hailo8", "@localhost")
#
#     # Specific models from configuration
#     face_rec = FaceRecognition.from_config(
#         hardware="hailo8",
#         detector_model="yolo_v8n_face_detection",
#         embedder_model="face_embedding_mobilenet"
#     )
#
#     # Full custom control
#     config = PipelineModelConfig(
#         detector_model="custom_detector",
#         embedder_model="custom_embedder",
#         zoo_url="https://my.zoo",
#         inference_host_address="192.168.1.100"
#     )
#     face_rec = FaceRecognition.custom(config)
#
#     # Enroll a new person
#     face_rec.enroll_person("john_doe", ["john1.jpg", "john2.jpg", "john3.jpg"])
#
#     # Verify identity (1:1)
#     is_match, confidence = face_rec.verify_person("test_image.jpg", "john_doe")
#
#     # Identify person (1:N)
#     person_id, confidence = face_rec.identify_person("unknown_face.jpg")
#

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass

from .face_detector import FaceDetector
from .face_embedder import FaceEmbedder
from .reid_database import ReID_Database
from .pipeline_config import PipelineModelConfig
from .face_utils import (
    face_align_and_crop,
    landmarks_from_dict,
    normalize_embedding,
    validate_landmarks,
)


class FaceRecognition:
    # ...existing code...

    def detect_faces(self, image_input):
        """
        Detect faces in the input image using the detector. Returns the PySDK results object.
        Accepts any input supported by the detector (path, URL, numpy array, PIL, etc).
        """
        return self.detector.detect(image_input)

    def align_faces(self, image_input, detections=None):
        """
        Align and crop all faces in the input image. Returns a list of aligned face images.
        If detections is None, will run detection first.
        image_input can be a path, URL, numpy array, or PySDK results object.
        """
        # If input is a PySDK results object, extract image and detections
        if hasattr(image_input, "image") and hasattr(image_input, "results"):
            image_obj = image_input.image
            detections = image_input.results if detections is None else detections
        else:
            # Otherwise, run detection
            results_obj = self.detect_faces(image_input)
            image_obj = results_obj.image
            detections = results_obj.results
        aligned = []
        for det in detections or []:
            landmarks = self._extract_landmarks(det)
            if landmarks is not None:
                face = self._align_face(image_obj, landmarks)
                if face is not None:
                    aligned.append(face)
        return aligned

    def get_face_embeddings(self, aligned_faces):
        """
        Get embeddings for a list of aligned face images. Returns a list of embedding vectors.
        """
        embeddings = []
        for face in aligned_faces:
            embedding_result = self.embedder.embed(face)
            if embedding_result is not None and embedding_result.results:
                embedding_data = embedding_result.results[0].get("data")
                if embedding_data is not None:
                    embeddings.append(normalize_embedding(np.array(embedding_data)))
        return embeddings

    def get_identities(self, detections, embeddings):
        """
        Patch the 'label' field in detections in-place using the provided embeddings.
        Returns the detections list with updated labels (for image_overlay compatibility).
        Now uses the score returned by get_attributes_by_embedding directly.
        """
        for det, embedding in zip(detections, embeddings):
            person_id, person_name, score = self.db.get_attributes_by_embedding(
                embedding
            )
            if (
                person_id
                and person_name
                and score is not None
                and score >= self.similarity_threshold
            ):
                det["label"] = person_name
                det["person_name"] = person_name
                det["similarity"] = score
                det["db_result"] = person_id
            else:
                det["label"] = "Unknown"
                det["person_name"] = None
                det["similarity"] = score if score is not None else 0.0
                det["db_result"] = person_id
        return detections

    def identify_faces(self, image_input):
        """
        High-level: detect faces, align, embed, and patch labels in detections.
        Returns detections list with 'label' and other fields for overlay.
        """
        results_obj = self.detect_faces(image_input)
        detections = results_obj.results
        if not detections:
            return []
        aligned = self.align_faces(results_obj, detections)
        if not aligned:
            return []
        embeddings = self.get_face_embeddings(aligned)
        if not embeddings:
            return []
        return self.get_identities(detections, embeddings)

    def _select_largest_face(self, detections):
        """Select the largest face by bbox_area from detections."""
        if not detections:
            return None
        return max(detections, key=lambda det: det.get("bbox_area", 0))

    def _extract_landmarks(self, detection):
        """Extract landmarks from a detection dict/object."""
        if hasattr(detection, "landmarks") and detection.landmarks:
            return detection.landmarks
        elif hasattr(detection, "keypoints") and detection.keypoints:
            return detection.keypoints
        elif hasattr(detection, "kpts") and detection.kpts:
            return detection.kpts
        elif isinstance(detection, dict):
            return (
                detection.get("landmarks")
                or detection.get("keypoints")
                or detection.get("kpts")
            )
        return None

    def _align_face(self, image_input, landmarks):
        """Align and crop face given image and landmarks."""
        if image_input is None or landmarks is None:
            return None
        landmarks_np = landmarks_from_dict(landmarks)
        return face_align_and_crop(image_input, landmarks_np, self.embedding_size)

    """
    Comprehensive face recognition pipeline with enrollment, verification, and identification.

    This class provides a unified interface for all face recognition operations,
    integrating face detection, embedding generation, and database management.
    It automatically handles model loading, face quality assessment, and
    database operations.

    The FaceRecognition class provides three ways to create a pipeline:

    1. Auto Mode (Default): Automatically selects the best available models for your hardware
        ```python
        face_rec = FaceRecognition.auto("hailo8")         # Auto mode - recommended
        face_rec = FaceRecognition.auto("hailo8", "@localhost")  # Local inference
        ```

    2. From Config: Use specific models from the built-in configuration
        ```python
        face_rec = FaceRecognition.from_config(
            hardware="hailo8",
            detector_model="yolo_v8n_face_detection",
            embedder_model="face_embedding_mobilenet"
        )
        ```

    3. Custom: Full control over model loading
        ```python
        config = PipelineModelConfig(
            detector_model="custom_detector_model",
            embedder_model="custom_embedder_model",
            zoo_url="https://custom.zoo.url",
            inference_host_address="192.168.1.100"
        )
        face_rec = FaceRecognition.custom(config)
        ```

    Key Features:
        - Auto-configured model selection based on hardware
        - Quality-controlled face enrollment with multiple photos
        - 1:1 verification with configurable thresholds
        - 1:N identification with ranking and confidence
        - Robust database management with deduplication
        - Comprehensive logging and error handling
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        *,
        db_path: str = "face_recognition.lance",
        similarity_threshold: float = 0.3,
        quality_threshold: float = 0.5,
        max_faces_per_image: int = 5,
        embedding_size: int = 112,
        enable_logging: bool = True,
    ):
        """
        Initialize the face recognition pipeline with pre-configured models.

        Note: This constructor is primarily for internal use. Users should prefer
        the factory methods: auto(), from_config(), or custom().

        Args:
            detector: Pre-configured FaceDetector instance
            embedder: Pre-configured FaceEmbedder instance
            db_path: Path to face database file (LanceDB format)
            similarity_threshold: Minimum similarity for face matching (0-1)
            quality_threshold: Minimum quality score for face enrollment (0-1)
            max_faces_per_image: Maximum faces to process per image
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If database cannot be initialized
        """

        # Configure logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Validate parameters
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if quality_threshold < 0 or quality_threshold > 1:
            raise ValueError("quality_threshold must be between 0 and 1")
        if max_faces_per_image < 1:
            raise ValueError("max_faces_per_image must be at least 1")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")

        # Store configuration
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold
        self.max_faces_per_image = max_faces_per_image
        self.embedding_size = embedding_size

        # Store model instances
        self.detector = detector
        self.embedder = embedder

        self.logger.info(
            "Face recognition pipeline initialized with pre-configured models"
        )

        # Initialize database
        try:
            self.db = ReID_Database(db_path, threshold=1.0 - similarity_threshold)
            self.logger.info(f"Database initialized: {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise ConnectionError(
                f"Could not initialize database at {db_path}: {e}"
            ) from e

    @classmethod
    def auto(
        cls,
        hardware: str,
        inference_host_address: str = "@cloud",
        *,
        db_path: str = "face_recognition.lance",
        similarity_threshold: float = 0.3,
        quality_threshold: float = 0.5,
        max_faces_per_image: int = 5,
        embedding_size: int = 112,
        enable_logging: bool = True,
    ):
        """
        Create a face recognition pipeline using automatic model selection.

        This method automatically selects the best available face detection
        and embedding models for the specified hardware.

        Args:
            hardware: AI hardware device ("hailo8", "cuda", "cpu", etc.)
            inference_host_address: Where to run inference
                - "@cloud": DeGirum cloud inference
                - "@localhost": Local inference
                - IP address: Remote inference on specific host
            db_path: Path to face database file
            similarity_threshold: Minimum similarity for face matching (0-1)
            quality_threshold: Minimum quality score for face enrollment (0-1)
            max_faces_per_image: Maximum faces to process per image
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Returns:
            FaceRecognition instance with automatically selected models

        Examples:
            >>> # Auto mode with cloud inference
            >>> face_rec = FaceRecognition.auto("hailo8")
            >>>
            >>> # Auto mode with local inference
            >>> face_rec = FaceRecognition.auto("hailo8", "@localhost")
            >>>
            >>> # Auto mode with custom settings
            >>> face_rec = FaceRecognition.auto(
            ...     hardware="cpu",
            ...     inference_host_address="@cloud",
            ...     similarity_threshold=0.7,
            ...     max_faces_per_image=3
            ... )
        """

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info(
            f"Initializing face recognition pipeline in auto mode for {hardware} "
            f"with inference on {inference_host_address}"
        )

        try:
            # Create detector and embedder using auto mode
            detector = FaceDetector.auto(hardware, inference_host_address)
            embedder = FaceEmbedder.auto(hardware, inference_host_address)

            logger.info("Models loaded successfully")

            return cls(
                detector=detector,
                embedder=embedder,
                db_path=db_path,
                similarity_threshold=similarity_threshold,
                quality_threshold=quality_threshold,
                max_faces_per_image=max_faces_per_image,
                embedding_size=embedding_size,
                enable_logging=enable_logging,
            )

        except Exception as e:
            logger.error(f"Failed to create auto mode pipeline: {e}")
            raise ConnectionError(
                f"Could not initialize face recognition pipeline in auto mode: {e}"
            ) from e

    @classmethod
    def from_config(
        cls,
        hardware: str,
        detector_model: str,
        embedder_model: str,
        inference_host_address: str = "@cloud",
        *,
        db_path: str = "face_recognition.lance",
        similarity_threshold: float = 0.6,
        quality_threshold: float = 0.5,
        max_faces_per_image: int = 5,
        embedding_size: int = 112,
        enable_logging: bool = True,
    ):
        """
        Create a face recognition pipeline using specific models from configuration.

        This method loads specific face detection and embedding models from
        the built-in model configuration for the specified hardware.

        Args:
            hardware: AI hardware device ("hailo8", "cuda", "cpu", etc.)
            detector_model: Face detector model name from configuration
            embedder_model: Face embedder model name from configuration
            inference_host_address: Where to run inference
            db_path: Path to face database file
            similarity_threshold: Minimum similarity for face matching (0-1)
            quality_threshold: Minimum quality score for face enrollment (0-1)
            max_faces_per_image: Maximum faces to process per image
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Returns:
            FaceRecognition instance with specified models from config

        Examples:
            >>> # Use specific models from config
            >>> face_rec = FaceRecognition.from_config(
            ...     hardware="hailo8",
            ...     detector_model="yolo_v8n_face_detection",
            ...     embedder_model="face_embedding_mobilenet"
            ... )
            >>>
            >>> # With remote inference
            >>> face_rec = FaceRecognition.from_config(
            ...     hardware="hailo8",
            ...     detector_model="yolo_v8n_face_detection",
            ...     embedder_model="face_embedding_mobilenet",
            ...     inference_host_address="192.168.1.100"
            ... )
        """

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info(
            f"Initializing face recognition pipeline from config for {hardware} "
            f"with models: {detector_model}, {embedder_model}"
        )

        try:
            # Create detector and embedder using from_config mode
            detector = FaceDetector.from_config(
                hardware, detector_model, inference_host_address
            )
            embedder = FaceEmbedder.from_config(
                hardware, embedder_model, inference_host_address
            )

            logger.info("Models loaded successfully from configuration")

            return cls(
                detector=detector,
                embedder=embedder,
                db_path=db_path,
                similarity_threshold=similarity_threshold,
                quality_threshold=quality_threshold,
                max_faces_per_image=max_faces_per_image,
                embedding_size=embedding_size,
                enable_logging=enable_logging,
            )

        except Exception as e:
            logger.error(f"Failed to create pipeline from config: {e}")
            raise ConnectionError(
                f"Could not initialize face recognition pipeline from config: {e}"
            ) from e

    @classmethod
    def custom(
        cls,
        config: PipelineModelConfig,
        *,
        db_path: str = "face_recognition.lance",
        similarity_threshold: float = 0.6,
        quality_threshold: float = 0.5,
        max_faces_per_image: int = 5,
        embedding_size: int = 112,
        enable_logging: bool = True,
    ):
        """
        Create a face recognition pipeline with full custom model control.

        This method provides complete control over model loading by accepting
        a PipelineModelConfig that specifies exact model names, zoo URLs,
        and inference locations.

        Args:
            config: PipelineModelConfig with custom model specifications
            db_path: Path to face database file
            similarity_threshold: Minimum similarity for face matching (0-1)
            quality_threshold: Minimum quality score for face enrollment (0-1)
            max_faces_per_image: Maximum faces to process per image
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Returns:
            FaceRecognition instance with fully custom models

        Examples:
            >>> # Full custom configuration
            >>> config = PipelineModelConfig(
            ...     detector_model="yolo_v8n_face_detection--512x512_quant_n2x_orca1_bgr",
            ...     embedder_model="face_embedding_mobilenet--112x112_quant_n2x_orca1_bgr",
            ...     zoo_url="https://cs.degirum.com/degirum/orca",
            ...     inference_host_address="192.168.1.100"
            ... )
            >>> face_rec = FaceRecognition.custom(config)
            >>>
            >>> # Cloud inference with custom models
            >>> config = PipelineModelConfig(
            ...     detector_model="custom_face_detector_v2",
            ...     embedder_model="custom_face_embedder_v2",
            ...     zoo_url="https://my.custom.zoo",
            ...     inference_host_address="@cloud"
            ... )
            >>> face_rec = FaceRecognition.custom(config, similarity_threshold=0.8)
        """

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info(
            f"Initializing face recognition pipeline with custom config:\n"
            f"  Detector: {config.detector_model}\n"
            f"  Embedder: {config.embedder_model}\n"
            f"  Zoo URL: {config.zoo_url}\n"
            f"  Inference: {config.inference_host_address}"
        )

        try:
            # Create detector and embedder using custom mode
            detector = FaceDetector.custom(
                config.detector_model, config.zoo_url, config.inference_host_address
            )
            embedder = FaceEmbedder.custom(
                config.embedder_model, config.zoo_url, config.inference_host_address
            )

            logger.info("Custom models loaded successfully")

            return cls(
                detector=detector,
                embedder=embedder,
                db_path=db_path,
                similarity_threshold=similarity_threshold,
                quality_threshold=quality_threshold,
                max_faces_per_image=max_faces_per_image,
                embedding_size=embedding_size,
                enable_logging=enable_logging,
            )

        except Exception as e:
            logger.error(f"Failed to create custom pipeline: {e}")
            raise ConnectionError(
                f"Could not initialize face recognition pipeline with custom config: {e}"
            ) from e

    def _get_largest_face(
        self, image_input
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get the largest face from an image by area.
        Args:
            image_input: Any input supported by the detector (path, URL, numpy array, PIL, etc)
        Returns:
            Tuple of (aligned_face, detection_info) or (None, None) if no face found
        """
        try:
            image_obj, detections = self._detect_faces(image_input)
            if not detections:
                return None, None
            if len(detections) > 1:
                self.logger.warning(
                    f"Multiple faces ({len(detections)}) found in image: {image_input}. Using largest face."
                )
            largest_face = self._select_largest_face(detections)
            if largest_face is None:
                return None, None
            landmarks_data = self._extract_landmarks(largest_face)
            if landmarks_data is None:
                self.logger.warning(
                    f"No landmarks found in detection for {image_input}"
                )
                return None, None
            aligned_face = self._align_face(image_obj, landmarks_data)
            return aligned_face, largest_face
        except Exception as e:
            self.logger.error(f"Error processing image {image_input}: {e}")
            return None, None

    def enroll(self, person_name: str, image_inputs: List):
        """
        Enroll a person in the database with multiple images.
        """
        try:
            self.logger.info(f"Starting enrollment for person: {person_name}")
            existing_person_id = self.db.get_id_by_attributes(person_name)
            if existing_person_id is not None:
                self.logger.info(
                    f"Person '{person_name}' already exists in database with ID {existing_person_id}. Adding new embeddings."
                )
                person_id = existing_person_id
            else:
                import uuid

                person_id = str(uuid.uuid4())
                self.logger.info(
                    f"Creating new person '{person_name}' with ID {person_id}"
                )
            embeddings = []
            valid_images = []
            # Ensure all image_inputs are strings or numpy arrays
            processed_inputs = []
            for img_input in image_inputs:
                if isinstance(img_input, Path):
                    processed_inputs.append(str(img_input))
                else:
                    processed_inputs.append(img_input)

            for img_input in processed_inputs:
                self.logger.info(f"Processing image: {img_input}")
                # Use the unified detect_faces method
                results_obj = self.detect_faces(img_input)
                image_obj = results_obj.image
                detections = results_obj.results
                if not detections:
                    self.logger.warning(
                        f"Skipping image {img_input} - no valid face found"
                    )
                    continue
                largest_face = self._select_largest_face(detections)
                if largest_face is None:
                    self.logger.warning(
                        f"Skipping image {img_input} - no face detected"
                    )
                    continue
                landmarks_data = self._extract_landmarks(largest_face)
                if landmarks_data is None:
                    self.logger.warning(
                        f"Skipping image {img_input} - no landmarks found"
                    )
                    continue
                aligned_face = self._align_face(image_obj, landmarks_data)
                if aligned_face is None:
                    self.logger.warning(
                        f"Skipping image {img_input} - could not align face"
                    )
                    continue
                embedding_result = self.embedder.embed(aligned_face)
                if embedding_result is not None and embedding_result.results:
                    embedding_data = embedding_result.results[0].get("data")
                    if embedding_data is not None:
                        normalized_embedding = normalize_embedding(
                            np.array(embedding_data)
                        )
                        embeddings.append(normalized_embedding)
                        valid_images.append(str(img_input))
                    else:
                        self.logger.warning(f"No embedding data found for {img_input}")
                else:
                    self.logger.warning(f"Failed to generate embedding for {img_input}")
            if not embeddings:
                self.logger.error(
                    f"No valid embeddings generated for person: {person_name}"
                )
                return False
            if existing_person_id is None:
                self.db.add_object(person_id, person_name)
                self.logger.info(
                    f"Added new person '{person_name}' to attributes table"
                )
            self.db.add_embeddings(person_id, embeddings)
            self.logger.info(
                f"Successfully enrolled person '{person_name}' with ID {person_id} using {len(embeddings)} images"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error enrolling person {person_name}: {e}")
            return False

    def verify(self, image1_input, image2_input) -> Tuple[bool, float]:
        """
        Verify if two images contain the same person.
        """
        try:
            self.logger.info(f"Verifying faces in {image1_input} and {image2_input}")
            # Get largest face from each image
            img1, detections1 = self._detect_faces(image1_input)
            img2, detections2 = self._detect_faces(image2_input)
            if not detections1 or not detections2:
                self.logger.error("Could not extract faces from one or both images")
                return False, 0.0
            largest1 = self._select_largest_face(detections1)
            largest2 = self._select_largest_face(detections2)
            if largest1 is None or largest2 is None:
                self.logger.error("Could not find largest face in one or both images")
                return False, 0.0
            landmarks1 = self._extract_landmarks(largest1)
            landmarks2 = self._extract_landmarks(largest2)
            if landmarks1 is None or landmarks2 is None:
                self.logger.error("Could not extract landmarks from one or both faces")
                return False, 0.0
            face1 = self._align_face(img1, landmarks1)
            face2 = self._align_face(img2, landmarks2)
            if face1 is None or face2 is None:
                self.logger.error("Could not align one or both faces")
                return False, 0.0
            embedding_result1 = self.embedder.embed(face1)
            embedding_result2 = self.embedder.embed(face2)
            if embedding_result1 is None or embedding_result2 is None:
                self.logger.error("Could not generate embeddings for one or both faces")
                return False, 0.0
            embedding1 = (
                embedding_result1.results[0].get("data")
                if embedding_result1.results
                else None
            )
            embedding2 = (
                embedding_result2.results[0].get("data")
                if embedding_result2.results
                else None
            )
            if embedding1 is None or embedding2 is None:
                self.logger.error("Could not extract embedding data from results")
                return False, 0.0
            norm_embedding1 = normalize_embedding(np.array(embedding1))
            norm_embedding2 = normalize_embedding(np.array(embedding2))
            similarity = np.dot(norm_embedding1, norm_embedding2)
            is_match = similarity >= self.similarity_threshold
            self.logger.info(
                f"Verification result: {'MATCH' if is_match else 'NO MATCH'} (similarity: {similarity:.3f})"
            )
            return is_match, float(similarity)
        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return False, 0.0
