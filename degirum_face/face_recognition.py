"""
face_recognition.py: Unified face recognition pipeline

Copyright DeGirum Corporation 2025

This module implements a clear, explicit face recognition pipeline:
    - Face detection
    - Face alignment
    - Face embedding
    - Database-backed identification and verification

Pipeline design principles:
    - Each step is explicit: detection, alignment, embedding, and identity assignment are separate and composable.
    - No hidden side effects: all methods require their true inputs (e.g., align_faces always takes image and detections).
    - Filtering is annotation-based: detections are never removed, only annotated (e.g., face_rejected, reject_reason).
    - All pipeline steps always receive the full list of detections and are robust to missing or repeated processing.
    - The pipeline is robust to user errors: steps can be run out of order or multiple times without crashing.
    - Model selection is hardware-aware and configurable.
    - Database management is robust and deduplicated.

Typical usage:
    from degirum_face import FaceRecognition
    face_rec = FaceRecognition.auto("hailo8")
    results = face_rec.detect_faces("image.jpg")
    aligned = face_rec.align_faces(results.image, results.results)
    embeddings = face_rec.get_face_embeddings(aligned)
    identities = face_rec.get_identities(results.results, embeddings)
    # Or use high-level identify_faces(image) for the full pipeline
    # Filtering is explicit and composable; all detections are always passed through the pipeline.
"""

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

# Optional: import default face filter utility
try:
    from .face_filters_integration import get_default_face_filter
except ImportError:
    get_default_face_filter = None


class FaceRecognition:
    # ...existing code...

    def filter_faces(self, detections):
        """
        Annotate each detection with face_rejected and reject_reason using self.face_filter.
        Filtering is annotation-based: detections are never removed, only marked as rejected.
        Always returns the full detections list (in-place modification).
        Downstream steps skip rejected faces but always receive the full list.
        This method is robust to being called multiple times or omitted entirely.
        """
        if self.face_filter is not None and detections:
            for det in detections:
                self.face_filter.filter_face_with_reason(det)
        return detections

    def detect_faces(self, image_input):
        """
        Detect faces in the input image using the detector. Returns the PySDK results object.
        Accepts any input supported by the detector (path, URL, numpy array, PIL, etc).
        No filtering is performed here; all detections are returned.
        """
        return self.detector.detect(image_input)

    def align_faces(self, image_obj, detections):
        """
        Align and crop all faces in the input image using provided detections.
        Skips detections with face_rejected=True (if present).
        Returns a list of (detection, aligned_face) tuples for non-rejected faces.
        Args:
            image_obj: The image (numpy array, etc.)
            detections: List of detection dicts/objects (all, including rejected)
        This method is robust to missing or repeated filtering; it only processes non-rejected faces.
        """
        aligned_pairs = []
        for detection in detections or []:
            if detection.get("face_rejected", False):
                continue
            landmarks = self._extract_landmarks(detection)
            if landmarks is not None:
                aligned_face = self._align_face(image_obj, landmarks)
                if aligned_face is not None:
                    aligned_pairs.append((detection, aligned_face))
        return aligned_pairs

    def get_face_embeddings(self, aligned_pairs):
        """
        Get embeddings for a list of (detection, aligned_face) pairs.
        Returns a list of (detection, embedding) pairs.
        This method is robust to missing or repeated alignment; it only processes aligned faces.
        """
        embedding_pairs = []
        for detection, aligned_face in aligned_pairs:
            embedding_result = self.embedder.embed(aligned_face)
            if embedding_result is not None and embedding_result.results:
                embedding_data = embedding_result.results[0].get("data")
                if embedding_data is not None:
                    embedding = normalize_embedding(np.array(embedding_data))
                    embedding_pairs.append((detection, embedding))
        return embedding_pairs

    def get_identities(self, detections, embedding_pairs):
        """
        Patch the 'label' field in detections in-place using the provided embeddings.
        Only updates detections that have embeddings (non-rejected faces).
        Returns the detections list with updated labels (for image_overlay compatibility).
        This method is robust to missing or repeated embedding; it only updates detections with embeddings.
        """
        for detection, embedding in embedding_pairs:
            person_id, person_name, score = self.db.get_attributes_by_embedding(
                embedding
            )
            if (
                person_id
                and person_name
                and score is not None
                and score >= self.similarity_threshold
            ):
                detection["label"] = person_name
                detection["person_name"] = person_name
                detection["similarity"] = score
                detection["db_result"] = person_id
            else:
                detection["label"] = "Unknown"
                detection["person_name"] = None
                detection["similarity"] = score if score is not None else 0.0
                detection["db_result"] = person_id
        return detections

    def identify_faces(self, image_input):
        """
        High-level: detect faces, filter, align, embed, and patch labels in detections.
        Returns the PySDK results object with labeled detections and overlay image.
        Filtering is explicit and annotation-based: detections are never removed, only marked as rejected.
        Downstream steps skip rejected faces but always receive the full list.
        The pipeline is robust to missing or repeated filtering, alignment, or embedding steps, and will not crash if steps are run out of order or multiple times.
        """
        results_obj = self.detect_faces(image_input)
        detections = results_obj.results
        image_obj = results_obj.image
        if not detections:
            return results_obj
        self.filter_faces(detections)
        aligned_pairs = self.align_faces(image_obj, detections)
        if not aligned_pairs:
            return results_obj
        embedding_pairs = self.get_face_embeddings(aligned_pairs)
        if not embedding_pairs:
            return results_obj
        self.get_identities(detections, embedding_pairs)  # patches in-place
        return results_obj

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
        embedding_size: int = 112,
        enable_logging: bool = True,
        face_filter=None,
    ):
        """
        Internal constructor for FaceRecognition. Use factory methods for user code.

        Args:
            detector: FaceDetector instance (already constructed)
            embedder: FaceEmbedder instance (already constructed)
            db_path: Path to face database file (LanceDB format)
            similarity_threshold: Minimum similarity for face matching (0-1)
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging
        """
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        self.similarity_threshold = similarity_threshold
        self.embedding_size = embedding_size
        self.detector = detector
        self.embedder = embedder
        self.face_filter = face_filter  # Optional FaceFilter instance
        self.logger.info(
            "Face recognition pipeline initialized with pre-configured models"
        )
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
        embedding_size: int = 112,
        enable_logging: bool = True,
        face_filter=None,
    ):
        """
        Create a face recognition pipeline using automatic model selection.

        Args:
            hardware: AI hardware device ("hailo8", "cuda", "cpu", etc.)
            inference_host_address: Where to run inference ("@cloud", "@localhost", or IP)
            db_path: Path to face database file
            similarity_threshold: Minimum similarity for face matching (0-1)
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Returns:
            FaceRecognition instance with automatically selected models
        """
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(
            f"Initializing face recognition pipeline in auto mode for {hardware} with inference on {inference_host_address}"
        )
        try:
            detector = FaceDetector.auto(hardware, inference_host_address)
            embedder = FaceEmbedder.auto(hardware, inference_host_address)
            logger.info("Models loaded successfully")
            return cls(
                detector=detector,
                embedder=embedder,
                db_path=db_path,
                similarity_threshold=similarity_threshold,
                embedding_size=embedding_size,
                enable_logging=enable_logging,
                face_filter=face_filter,
            )
        except Exception as e:
            logger.error(f"Failed to create auto mode pipeline: {e}")
            raise ConnectionError(
                f"Could not initialize face recognition pipeline in auto mode: {e}"
            ) from e

    @classmethod
    def from_config(
        cls,
        detector_model: str,
        embedder_model: str,
        inference_host_address: str = "@cloud",
        *,
        db_path: str = "face_recognition.lance",
        similarity_threshold: float = 0.6,
        embedding_size: int = 112,
        enable_logging: bool = True,
    ):
        """
        Create a face recognition pipeline using specific models from configuration.

        Args:
            detector_model: Face detector model name from configuration
            embedder_model: Face embedder model name from configuration
            inference_host_address: Where to run inference
            db_path: Path to face database file
            similarity_threshold: Minimum similarity for face matching (0-1)
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Returns:
            FaceRecognition instance with specified models from config
        """
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(
            f"Initializing face recognition pipeline from config with models: {detector_model}, {embedder_model}"
        )
        try:
            detector = FaceDetector.from_config(detector_model, inference_host_address)
            embedder = FaceEmbedder.from_config(embedder_model, inference_host_address)
            logger.info("Models loaded successfully from configuration")
            return cls(
                detector=detector,
                embedder=embedder,
                db_path=db_path,
                similarity_threshold=similarity_threshold,
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
        embedding_size: int = 112,
        enable_logging: bool = True,
    ):
        """
        Create a face recognition pipeline with full custom model control.

        Args:
            config: PipelineModelConfig with custom model specifications
            db_path: Path to face database file
            similarity_threshold: Minimum similarity for face matching (0-1)
            embedding_size: Target size for face alignment (112 recommended)
            enable_logging: Enable detailed logging

        Returns:
            FaceRecognition instance with fully custom models
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
