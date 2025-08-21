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


@dataclass
@dataclass
class FaceQualityMetrics:
    """Face quality assessment metrics for enrollment filtering."""

    face_size: float  # Face bounding box area
    confidence: float  # Detection confidence score
    landmark_quality: float  # Landmark detection quality
    frontal_score: float  # How frontal is the face (0-1)
    sharpness: float  # Image sharpness score
    brightness: float  # Face region brightness
    overall_quality: float  # Combined quality score (0-1)


@dataclass
class RecognitionResult:
    """Result of face recognition operation."""

    person_id: Optional[str]  # Identified person ID (None if unknown)
    confidence: float  # Recognition confidence (0-1)
    embedding: np.ndarray  # Face embedding vector
    quality: FaceQualityMetrics  # Face quality assessment
    bbox: Optional[List[float]] = None  # Face bounding box [x, y, w, h]
    landmarks: Optional[List[np.ndarray]] = None  # Facial landmarks


@dataclass
class EnrollmentResult:
    """Result of face enrollment operation."""

    person_id: str  # Person identifier
    num_faces_processed: int  # Total faces processed
    num_faces_enrolled: int  # Faces that passed quality checks
    num_faces_rejected: int  # Faces rejected due to quality
    quality_scores: List[float]  # Quality scores for each face
    embedding_count: int  # Total embeddings stored in database


class FaceRecognition:
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

    def enroll_person(
        self,
        person_id: str,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        *,
        attributes: Optional[Dict[str, Any]] = None,
        replace_existing: bool = False,
        min_faces_required: int = 1,
        max_faces_to_enroll: int = 10,
    ) -> EnrollmentResult:
        """
        Enroll a person in the face recognition database.

        This method processes multiple images of a person, extracts high-quality
        face embeddings, and stores them in the database for future recognition.
        It automatically handles quality filtering, deduplication, and database
        management.

        Args:
            person_id: Unique identifier for the person
            images: Single image or list of images (file paths, URLs, or numpy arrays)
            attributes: Optional person attributes (name, department, etc.)
            replace_existing: If True, replace existing person data
            min_faces_required: Minimum faces needed for successful enrollment
            max_faces_to_enroll: Maximum faces to store per person

        Returns:
            EnrollmentResult with enrollment statistics and quality metrics

        Raises:
            ValueError: If person_id is invalid or insufficient quality faces
            FileNotFoundError: If image files don't exist

        Examples:
            >>> # Enroll with multiple photos
            >>> result = face_rec.enroll_person(
            ...     "emp_001",
            ...     ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
            ...     attributes={"name": "John Doe", "role": "Manager"}
            ... )
            >>> print(f"Enrolled {result.num_faces_enrolled} faces")
            >>>
            >>> # Enroll from numpy arrays
            >>> face_images = [cv2.imread(f) for f in image_files]
            >>> result = face_rec.enroll_person("emp_002", face_images)
        """

        if not person_id or not isinstance(person_id, str):
            raise ValueError("person_id must be a non-empty string")

        # Convert single image to list
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image must be provided")

        self.logger.info(
            f"Starting enrollment for {person_id} with {len(images)} images"
        )

        # Check if person already exists
        if not replace_existing:
            existing_attrs = self.db.get_attributes_by_id(person_id)
            if existing_attrs is not None:
                raise ValueError(
                    f"Person {person_id} already exists. Use replace_existing=True to update."
                )

        # Process each image and extract embeddings
        all_embeddings = []
        quality_scores = []
        faces_processed = 0
        faces_rejected = 0

        for i, image in enumerate(images):
            try:
                self.logger.debug(
                    f"Processing image {i+1}/{len(images)} for {person_id}"
                )

                # Detect faces in image
                detection_result = self.detector.detect(image)

                if not detection_result.results:
                    self.logger.warning(f"No faces detected in image {i+1}")
                    faces_rejected += 1
                    continue

                # Process each detected face (up to max_faces_per_image)
                for j, face_det in enumerate(
                    detection_result.results[: self.max_faces_per_image]
                ):
                    faces_processed += 1

                    try:
                        # Extract face embedding and quality metrics
                        embedding, quality = self._extract_face_embedding(
                            image, face_det
                        )

                        if quality.overall_quality >= self.quality_threshold:
                            all_embeddings.append(embedding)
                            quality_scores.append(quality.overall_quality)
                            self.logger.debug(
                                f"Face {j+1} accepted with quality {quality.overall_quality:.3f}"
                            )
                        else:
                            faces_rejected += 1
                            self.logger.debug(
                                f"Face {j+1} rejected with quality {quality.overall_quality:.3f}"
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to process face {j+1} in image {i+1}: {e}"
                        )
                        faces_rejected += 1

            except Exception as e:
                self.logger.warning(f"Failed to process image {i+1}: {e}")
                faces_rejected += 1

        # Validate enrollment requirements
        if len(all_embeddings) < min_faces_required:
            raise ValueError(
                f"Insufficient quality faces for enrollment. "
                f"Required: {min_faces_required}, Found: {len(all_embeddings)}"
            )

        # Limit number of embeddings to store
        if len(all_embeddings) > max_faces_to_enroll:
            # Keep the highest quality embeddings
            sorted_indices = np.argsort(quality_scores)[::-1][:max_faces_to_enroll]
            all_embeddings = [all_embeddings[i] for i in sorted_indices]
            quality_scores = [quality_scores[i] for i in sorted_indices]

        # Store person attributes in database
        if attributes is None:
            attributes = {"enrolled_date": str(np.datetime64("now"))}
        else:
            attributes = dict(attributes)  # Make a copy
            attributes["enrolled_date"] = str(np.datetime64("now"))

        # Add person to database
        if replace_existing:
            # Remove existing embeddings first
            try:
                # TODO: Add method to remove person from database
                pass
            except:
                pass

        self.db.add_object(person_id, attributes)

        # Add embeddings to database with deduplication
        self.db.add_embeddings(person_id, all_embeddings, dedup=True)

        faces_enrolled = len(all_embeddings)

        self.logger.info(
            f"Enrollment complete for {person_id}: "
            f"{faces_enrolled} faces enrolled, {faces_rejected} rejected"
        )

        return EnrollmentResult(
            person_id=person_id,
            num_faces_processed=faces_processed,
            num_faces_enrolled=faces_enrolled,
            num_faces_rejected=faces_rejected,
            quality_scores=quality_scores,
            embedding_count=len(all_embeddings),
        )

    def verify_person(
        self,
        image: Union[str, np.ndarray],
        person_id: str,
        *,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Verify if a face image matches a specific person (1:1 verification).

        This method compares a face in the given image against the stored
        embeddings for a specific person and returns whether they match
        above the confidence threshold.

        Args:
            image: Image containing face (file path, URL, or numpy array)
            person_id: ID of person to verify against
            confidence_threshold: Override default similarity threshold

        Returns:
            Tuple of (is_match: bool, confidence: float)
            - is_match: True if face matches person above threshold
            - confidence: Similarity score (0-1, higher = more similar)

        Raises:
            ValueError: If person_id not found in database
            RuntimeError: If no face detected in image

        Examples:
            >>> # Basic verification
            >>> is_match, confidence = face_rec.verify_person("test.jpg", "john_doe")
            >>> if is_match:
            ...     print(f"Verified John with {confidence:.3f} confidence")
            >>>
            >>> # With custom threshold
            >>> is_match, confidence = face_rec.verify_person(
            ...     "test.jpg", "john_doe", confidence_threshold=0.8
            ... )
        """

        if confidence_threshold is None:
            confidence_threshold = self.similarity_threshold

        # Check if person exists in database
        person_attrs = self.db.get_attributes_by_id(person_id)
        if person_attrs is None:
            raise ValueError(f"Person {person_id} not found in database")

        self.logger.debug(f"Verifying image against person {person_id}")

        # Extract face embedding from image
        try:
            recognition_result = self._recognize_face(image)
            if recognition_result.person_id is None:
                # No face detected or quality too low
                return False, 0.0

        except Exception as e:
            self.logger.error(f"Failed to extract face from verification image: {e}")
            raise RuntimeError(f"Could not process verification image: {e}") from e

        # Get stored embeddings for the person and calculate similarity directly
        try:
            # Get all embeddings for the specific person
            person_embeddings = self._get_person_embeddings(person_id)
            if not person_embeddings:
                raise ValueError(f"No embeddings found for person {person_id}")

            # Calculate similarity against all embeddings for this person
            similarities = []
            from .face_utils import normalize_embedding

            test_embedding_norm = normalize_embedding(recognition_result.embedding)

            for stored_embedding in person_embeddings:
                stored_embedding_norm = normalize_embedding(stored_embedding)
                # Calculate cosine similarity
                similarity = float(np.dot(test_embedding_norm, stored_embedding_norm))
                similarities.append(similarity)

            # Use the best (highest) similarity score
            confidence = max(similarities) if similarities else 0.0
            is_match = confidence >= confidence_threshold

            self.logger.debug(
                f"Person-specific verification: {person_id} - similarity {confidence:.3f} vs threshold {confidence_threshold:.3f}"
            )
            return is_match, confidence

        except Exception as e:
            self.logger.error(f"Person-specific verification failed: {e}")
            return False, 0.0

    def identify_person(
        self,
        image: Union[str, np.ndarray],
        *,
        confidence_threshold: Optional[float] = None,
        return_top_n: int = 1,
    ) -> Union[Tuple[Optional[str], float], List[Tuple[str, float]]]:
        """
        Identify a person from a face image (1:N identification).

        This method searches the entire database to find the best matching
        person for a face in the given image. It can return the single best
        match or the top N candidates with confidence scores.

        Args:
            image: Image containing face (file path, URL, or numpy array)
            confidence_threshold: Override default similarity threshold
            return_top_n: Number of top candidates to return (1 for single result)

        Returns:
            If return_top_n=1: Tuple of (person_id: Optional[str], confidence: float)
            If return_top_n>1: List of (person_id: str, confidence: float) tuples

            person_id is None if no match above threshold found

        Raises:
            RuntimeError: If no face detected in image

        Examples:
            >>> # Single best match
            >>> person_id, confidence = face_rec.identify_person("unknown.jpg")
            >>> if person_id:
            ...     print(f"Identified as {person_id} with {confidence:.3f} confidence")
            ... else:
            ...     print("Unknown person")
            >>>
            >>> # Top 3 candidates
            >>> candidates = face_rec.identify_person("unknown.jpg", return_top_n=3)
            >>> for person_id, confidence in candidates:
            ...     print(f"{person_id}: {confidence:.3f}")
        """

        if confidence_threshold is None:
            confidence_threshold = self.similarity_threshold

        if return_top_n < 1:
            raise ValueError("return_top_n must be at least 1")

        self.logger.debug(f"Identifying person in image (top {return_top_n})")

        # Extract face embedding from image
        try:
            recognition_result = self._recognize_face(image)
            if recognition_result.person_id is None:
                # No face detected or quality too low
                if return_top_n == 1:
                    return None, 0.0
                else:
                    return []

        except Exception as e:
            self.logger.error(f"Failed to extract face from identification image: {e}")
            raise RuntimeError(f"Could not process identification image: {e}") from e

        # Search database for matches
        try:
            if return_top_n == 1:
                # Single best match - use original method for compatibility
                found_person_id, found_attrs, similarity = (
                    self.db.get_attributes_by_embedding(recognition_result.embedding)
                )

                if found_person_id and similarity >= confidence_threshold:
                    self.logger.debug(
                        f"Identified as {found_person_id} with {similarity:.3f} confidence"
                    )
                    return found_person_id, similarity
                else:
                    # No match above threshold
                    return None, (similarity if found_person_id else 0.0)
            else:
                # Multiple candidates - use new method
                matches = self.db.get_top_matches_by_embedding(
                    recognition_result.embedding, top_n=return_top_n
                )

                # Filter by confidence threshold
                valid_matches = [
                    (person_id, similarity)
                    for person_id, attrs, similarity in matches
                    if similarity >= confidence_threshold
                ]

                if valid_matches:
                    self.logger.debug(
                        f"Found {len(valid_matches)} matches above threshold"
                    )
                    return valid_matches
                else:
                    # No matches above threshold, but return best match with actual similarity
                    if matches:
                        best_person_id, _, best_similarity = matches[0]
                        self.logger.debug(
                            f"No matches above threshold, best was {best_person_id} with {best_similarity:.3f}"
                        )
                    return []

        except Exception as e:
            self.logger.error(f"Database search failed during identification: {e}")
            if return_top_n == 1:
                return None, 0.0
            else:
                return []

    def identify_all_persons(
        self,
        image: Union[str, np.ndarray],
        *,
        confidence_threshold: Optional[float] = None,
        return_top_n_per_face: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Identify all persons in an image (multiple face identification).

        This method detects all faces in the image and attempts to identify
        each one separately. Useful for group photos or multi-person scenarios.

        Args:
            image: Image containing faces (file path, URL, or numpy array)
            confidence_threshold: Override default similarity threshold
            return_top_n_per_face: Number of candidates to return per face

        Returns:
            List of dictionaries, one per detected face, containing:
            - face_index: Index of face in image (0, 1, 2, ...)
            - bbox: Face bounding box [x, y, width, height]
            - quality: Face quality metrics
            - person_id: Identified person ID (None if unknown)
            - confidence: Recognition confidence (0-1)
            - candidates: List of top candidates if return_top_n_per_face > 1

        Examples:
            >>> # Identify all faces in group photo
            >>> results = face_rec.identify_all_persons("group_photo.jpg")
            >>> for i, result in enumerate(results):
            ...     if result['person_id']:
            ...         print(f"Face {i}: {result['person_id']} ({result['confidence']:.3f})")
            ...     else:
            ...         print(f"Face {i}: Unknown person")
            >>>
            >>> # Get top 3 candidates for each face
            >>> results = face_rec.identify_all_persons("group.jpg", return_top_n_per_face=3)
            >>> for result in results:
            ...     print(f"Face {result['face_index']} candidates:")
            ...     for person_id, conf in result['candidates']:
            ...         print(f"  {person_id}: {conf:.3f}")
        """

        if confidence_threshold is None:
            confidence_threshold = self.similarity_threshold

        if return_top_n_per_face < 1:
            raise ValueError("return_top_n_per_face must be at least 1")

        self.logger.debug(f"Identifying all persons in image")

        # Detect all faces in image
        try:
            detection_result = self.detector.detect(image)

            if not detection_result.results:
                self.logger.debug("No faces detected in image")
                return []

        except Exception as e:
            self.logger.error(f"Failed to detect faces: {e}")
            raise RuntimeError(
                f"Could not process image for face detection: {e}"
            ) from e

        # Process each detected face
        all_results = []

        for face_index, face_det in enumerate(detection_result.results):
            try:
                self.logger.debug(
                    f"Processing face {face_index + 1}/{len(detection_result.results)}"
                )

                # Extract face embedding and quality
                embedding, quality = self._extract_face_embedding(image, face_det)

                # Check if face meets quality threshold
                if quality.overall_quality < self.quality_threshold:
                    self.logger.debug(
                        f"Face {face_index} rejected due to low quality: {quality.overall_quality:.3f}"
                    )

                    # Still include in results but mark as low quality
                    bbox = face_det.get("bbox") or face_det.get("box", [0, 0, 0, 0])
                    face_result = {
                        "face_index": face_index,
                        "bbox": [
                            bbox[0],
                            bbox[1],
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                        ],
                        "quality": quality,
                        "person_id": None,
                        "confidence": 0.0,
                        "candidates": [],
                        "rejection_reason": "low_quality",
                    }
                    all_results.append(face_result)
                    continue

                # Search database for matches
                try:
                    found_person_id, found_attrs, similarity = (
                        self.db.get_attributes_by_embedding(embedding)
                    )

                    # Use actual similarity score from database
                    confidence = similarity if found_person_id else 0.0

                    # Prepare face result
                    bbox = face_det.get("bbox") or face_det.get("box", [0, 0, 0, 0])
                    landmarks = (
                        face_det.get("landmarks")
                        or face_det.get("keypoints")
                        or face_det.get("kpts")
                    )
                    face_result = {
                        "face_index": face_index,
                        "bbox": [
                            bbox[0],
                            bbox[1],
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                        ],
                        "quality": quality,
                        "landmarks": (
                            landmarks_from_dict(landmarks) if landmarks else None
                        ),
                    }

                    if found_person_id and confidence >= confidence_threshold:
                        # Person identified above threshold
                        face_result.update(
                            {
                                "person_id": found_person_id,
                                "confidence": confidence,
                                "candidates": (
                                    [(found_person_id, confidence)]
                                    if return_top_n_per_face > 1
                                    else []
                                ),
                            }
                        )

                        self.logger.debug(
                            f"Face {face_index} identified as {found_person_id} (confidence: {confidence:.3f})"
                        )

                    else:
                        # No match above threshold
                        face_result.update(
                            {
                                "person_id": None,
                                "confidence": confidence,
                                "candidates": [],
                            }
                        )

                        self.logger.debug(
                            f"Face {face_index} not identified (confidence: {confidence:.3f})"
                        )

                    all_results.append(face_result)

                except Exception as e:
                    self.logger.warning(
                        f"Database search failed for face {face_index}: {e}"
                    )

                    # Add result indicating database error
                    bbox = face_det.get("bbox") or face_det.get("box", [0, 0, 0, 0])
                    face_result = {
                        "face_index": face_index,
                        "bbox": [
                            bbox[0],
                            bbox[1],
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                        ],
                        "quality": quality,
                        "person_id": None,
                        "confidence": 0.0,
                        "candidates": [],
                        "error": str(e),
                    }
                    all_results.append(face_result)

            except Exception as e:
                self.logger.warning(f"Failed to process face {face_index}: {e}")

                # Add result indicating processing error
                bbox = face_det.get("bbox") or face_det.get("box", [0, 0, 0, 0])
                face_result = {
                    "face_index": face_index,
                    "bbox": [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ],
                    "quality": None,
                    "person_id": None,
                    "confidence": 0.0,
                    "candidates": [],
                    "error": str(e),
                }
                all_results.append(face_result)

        self.logger.info(f"Processed {len(all_results)} faces in image")
        return all_results

    def get_enrolled_persons(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of all enrolled persons and their attributes.

        Returns:
            Dictionary mapping person_id to person attributes

        Example:
            >>> persons = face_rec.get_enrolled_persons()
            >>> for person_id, attrs in persons.items():
            ...     print(f"{person_id}: {attrs.get('name', 'Unknown')}")
        """
        return self.db.list_objects()

    def remove_person(self, person_id: str) -> bool:
        """
        Remove a person and all their embeddings from the database.

        Args:
            person_id: ID of person to remove

        Returns:
            True if person was removed, False if not found

        Note:
            This method will be implemented when database supports deletion
        """
        # TODO: Implement when ReID_Database supports person removal
        self.logger.warning("Person removal not yet implemented in database")
        return False

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the face database.

        Returns:
            Dictionary with database statistics including person count,
            embedding count, and storage information
        """
        try:
            embedding_counts = self.db.count_embeddings()
            total_embeddings = sum(count for count, _ in embedding_counts.values())

            return {
                "total_persons": len(embedding_counts),
                "total_embeddings": total_embeddings,
                "avg_embeddings_per_person": (
                    total_embeddings / len(embedding_counts) if embedding_counts else 0
                ),
                "persons_with_embeddings": embedding_counts,
            }
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

    def _recognize_face(self, image: Union[str, np.ndarray]) -> RecognitionResult:
        """
        Internal method to extract face embedding and perform recognition.

        Args:
            image: Input image

        Returns:
            RecognitionResult with embedding and recognition info

        Raises:
            RuntimeError: If no suitable face found
        """

        # Detect faces in image
        detection_result = self.detector.detect(image)

        if not detection_result.results:
            raise RuntimeError("No faces detected in image")

        # Find the best quality face
        best_face = None
        best_quality = 0
        best_embedding = None

        for face_det in detection_result.results[: self.max_faces_per_image]:
            try:
                embedding, quality = self._extract_face_embedding(image, face_det)

                if quality.overall_quality > best_quality:
                    best_quality = quality.overall_quality
                    best_face = face_det
                    best_embedding = embedding

            except Exception as e:
                self.logger.debug(f"Failed to process detected face: {e}")
                continue

        if best_embedding is None or best_quality < self.quality_threshold:
            raise RuntimeError(
                f"No high-quality face found (best quality: {best_quality:.3f})"
            )

        # Search database for match
        try:
            found_person_id, found_attrs, similarity = (
                self.db.get_attributes_by_embedding(best_embedding)
            )

            # Use actual similarity score from database
            confidence = similarity if found_person_id else 0.0

            bbox = best_face.get("bbox") or best_face.get("box", [0, 0, 0, 0])
            landmarks = (
                best_face.get("landmarks")
                or best_face.get("keypoints")
                or best_face.get("kpts")
            )
            return RecognitionResult(
                person_id=found_person_id,
                confidence=confidence,
                embedding=best_embedding,
                quality=self._assess_face_quality(image, best_face),
                bbox=[
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ],
                landmarks=(landmarks_from_dict(landmarks) if landmarks else None),
            )

        except Exception as e:
            self.logger.error(f"Database search failed: {e}")
            # Return result without database match
            bbox = best_face.get("bbox") or best_face.get("box", [0, 0, 0, 0])
            landmarks = (
                best_face.get("landmarks")
                or best_face.get("keypoints")
                or best_face.get("kpts")
            )
            return RecognitionResult(
                person_id=None,
                confidence=0.0,
                embedding=best_embedding,
                quality=self._assess_face_quality(image, best_face),
                bbox=[
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ],
                landmarks=(landmarks_from_dict(landmarks) if landmarks else None),
            )

    def _extract_face_embedding(
        self, image: Union[str, np.ndarray], face_detection: Any
    ) -> Tuple[np.ndarray, FaceQualityMetrics]:
        """
        Extract embedding from a detected face with quality assessment.

        Args:
            image: Original image
            face_detection: Face detection result

        Returns:
            Tuple of (embedding, quality_metrics)
        """

        # Load image if needed
        if isinstance(image, str):
            import cv2

            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img_array = image

        # Extract landmarks and align face
        # face_detection is a dictionary, not an object
        landmarks_data = None

        # Try different possible keys for landmarks/keypoints
        if isinstance(face_detection, dict):
            landmarks_data = (
                face_detection.get("landmarks")
                or face_detection.get("keypoints")
                or face_detection.get("kpts")
            )
        else:
            # Fallback: try as object attributes (legacy support)
            if hasattr(face_detection, "landmarks") and face_detection.landmarks:
                landmarks_data = face_detection.landmarks
            elif hasattr(face_detection, "keypoints") and face_detection.keypoints:
                landmarks_data = face_detection.keypoints
            elif hasattr(face_detection, "kpts") and face_detection.kpts:
                landmarks_data = face_detection.kpts

        if landmarks_data is not None:
            landmarks = landmarks_from_dict(landmarks_data)

            # Align and crop face
            aligned_face = face_align_and_crop(
                img_array, landmarks, self.embedding_size
            )

            # Generate embedding
            embedding_result = self.embedder.embed(aligned_face)
            embedding = embedding_result.results[0].get("data")

            # Normalize embedding
            normalized_embedding = normalize_embedding(embedding)

            # Assess face quality
            quality = self._assess_face_quality(img_array, face_detection)

            return normalized_embedding, quality
        else:
            raise ValueError("No landmarks found for face alignment")

    def _assess_face_quality(
        self, image: np.ndarray, face_detection: Any
    ) -> FaceQualityMetrics:
        """
        Assess the quality of a detected face for enrollment/recognition.

        Args:
            image: Original image
            face_detection: Face detection result (dict or object)

        Returns:
            FaceQualityMetrics with quality scores
        """

        # Extract basic metrics - handle both dict and object formats
        if isinstance(face_detection, dict):
            bbox = face_detection.get("bbox", [0, 0, 100, 100])
            confidence = face_detection.get("confidence", 0.5)
            landmarks_data = (
                face_detection.get("landmarks")
                or face_detection.get("keypoints")
                or face_detection.get("kpts")
            )
        else:
            # Legacy object format
            bbox = getattr(face_detection, "bbox", [0, 0, 100, 100])
            confidence = getattr(face_detection, "confidence", 0.5)
            landmarks_data = getattr(face_detection, "landmarks", None)

        face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Calculate landmark quality
        landmark_quality = 1.0
        if landmarks_data is not None:
            try:
                landmarks = landmarks_from_dict(landmarks_data)
                landmark_quality = 1.0 if validate_landmarks(landmarks) else 0.0
            except:
                landmark_quality = 0.0
        else:
            landmark_quality = 0.0

        # Estimate frontal score (placeholder - would need pose estimation)
        frontal_score = 0.8  # TODO: Implement actual pose estimation

        # Estimate sharpness (placeholder - would need blur detection)
        sharpness = 0.8  # TODO: Implement blur detection

        # Estimate brightness (placeholder - would need exposure analysis)
        brightness = 0.8  # TODO: Implement exposure analysis

        # Calculate overall quality score
        overall_quality = (
            min(face_size / (100 * 100), 1.0) * 0.2  # Face size factor
            + confidence * 0.3  # Detection confidence
            + landmark_quality * 0.2  # Landmark quality
            + frontal_score * 0.1  # Frontal pose
            + sharpness * 0.1  # Image sharpness
            + brightness * 0.1  # Lighting quality
        )

        return FaceQualityMetrics(
            face_size=face_size,
            confidence=confidence,
            landmark_quality=landmark_quality,
            frontal_score=frontal_score,
            sharpness=sharpness,
            brightness=brightness,
            overall_quality=overall_quality,
        )

    def _get_person_embeddings(self, person_id: str) -> List[np.ndarray]:
        """
        Get all stored embeddings for a specific person.

        Args:
            person_id: The person identifier

        Returns:
            List of embedding vectors for the person
        """
        embeddings = []

        try:
            # Access the embeddings table directly
            embeddings_table, _ = self.db._open_table(self.db.tbl_embeddings)
            if embeddings_table is None:
                return embeddings

            # Query for all embeddings for this person
            person_embeddings = (
                embeddings_table.search()
                .where(f"{self.db.key_object_id} == '{person_id}'")
                .to_list()
            )

            for emb_record in person_embeddings:
                embedding_data = emb_record[self.db.key_embedding]
                if isinstance(embedding_data, list):
                    embedding = np.array(embedding_data, dtype=np.float32)
                else:
                    embedding = np.array(embedding_data, dtype=np.float32)
                embeddings.append(embedding)

        except Exception as e:
            self.logger.error(f"Failed to retrieve embeddings for {person_id}: {e}")

        return embeddings
