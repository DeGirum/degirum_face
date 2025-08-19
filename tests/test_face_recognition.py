#!/usr/bin/env python3
"""
Unit tests for Face Recognition Pipeline

Tests the core functionality of the face recognition system including
enrollment, verification, identification, and database management.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add the degirum_face package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from degirum_face import (
    FaceRecognition,
    EnrollmentResult,
    RecognitionResult,
    FaceQualityMetrics,
)


class MockFaceDetector:
    """Mock face detector for testing."""

    def detect(self, image):
        """Mock detection that returns a fake face detection result."""

        class MockDetectionResult:
            def __init__(self):
                self.results = [MockFaceDetection()]

        return MockDetectionResult()


class MockFaceDetection:
    """Mock face detection result."""

    def __init__(self):
        self.bbox = [100, 100, 200, 200]  # x1, y1, x2, y2
        self.confidence = 0.95
        self.landmarks = [
            {"landmark": [120, 130]},  # left eye
            {"landmark": [180, 130]},  # right eye
            {"landmark": [150, 150]},  # nose
            {"landmark": [130, 170]},  # left mouth
            {"landmark": [170, 170]},  # right mouth
        ]


class MockFaceEmbedder:
    """Mock face embedder for testing."""

    def embed(self, image):
        """Mock embedding that returns a fake embedding result."""

        class MockEmbeddingResult:
            def __init__(self):
                # Return a random but consistent embedding
                np.random.seed(42)  # Consistent for testing
                self.results = [{"data": np.random.randn(512).astype(np.float32)}]

        return MockEmbeddingResult()


class TestFaceRecognitionPipeline(unittest.TestCase):
    """Test cases for the FaceRecognition pipeline."""

    def setUp(self):
        """Set up test fixtures before each test method."""

        # Create temporary database directory
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_faces.lance")

        # Create test images directory
        self.test_images_dir = Path(self.temp_dir) / "test_images"
        self.test_images_dir.mkdir()

        # Create mock test images
        self._create_mock_images()

    def tearDown(self):
        """Clean up after each test method."""

        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_images(self):
        """Create mock image files for testing."""

        import cv2

        # Create simple test images
        for person in ["alice", "bob", "carol"]:
            person_dir = self.test_images_dir / person
            person_dir.mkdir()

            for i in range(3):
                img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
                cv2.putText(
                    img,
                    f"{person}_{i}",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imwrite(str(person_dir / f"{person}_{i}.jpg"), img)

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_face_recognition_initialization(self):
        """Test FaceRecognition initialization with different configurations."""

        # Test auto-initialization
        face_rec = FaceRecognition(hardware=None, db_path=self.db_path)

        self.assertIsNotNone(face_rec.detector)
        self.assertIsNotNone(face_rec.embedder)
        self.assertIsNotNone(face_rec.db)

        # Test with custom parameters
        face_rec_custom = FaceRecognition(
            hardware="cpu",
            db_path=self.db_path,
            similarity_threshold=0.7,
            quality_threshold=0.6,
            max_faces_per_image=2,
        )

        self.assertEqual(face_rec_custom.similarity_threshold, 0.7)
        self.assertEqual(face_rec_custom.quality_threshold, 0.6)
        self.assertEqual(face_rec_custom.max_faces_per_image, 2)

    def test_face_recognition_parameter_validation(self):
        """Test parameter validation during initialization."""

        # Test invalid similarity threshold
        with self.assertRaises(ValueError):
            FaceRecognition(db_path=self.db_path, similarity_threshold=1.5)

        with self.assertRaises(ValueError):
            FaceRecognition(db_path=self.db_path, similarity_threshold=-0.1)

        # Test invalid quality threshold
        with self.assertRaises(ValueError):
            FaceRecognition(db_path=self.db_path, quality_threshold=1.5)

        # Test invalid max faces per image
        with self.assertRaises(ValueError):
            FaceRecognition(db_path=self.db_path, max_faces_per_image=0)

        # Test invalid embedding size
        with self.assertRaises(ValueError):
            FaceRecognition(db_path=self.db_path, embedding_size=-1)

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_person_enrollment(self):
        """Test person enrollment with multiple images."""

        face_rec = FaceRecognition(hardware="cpu", db_path=self.db_path)

        # Test successful enrollment
        alice_images = list((self.test_images_dir / "alice").glob("*.jpg"))
        alice_images = [str(img) for img in alice_images]

        result = face_rec.enroll_person(
            person_id="alice_test",
            images=alice_images,
            attributes={"name": "Alice Test", "department": "Engineering"},
        )

        self.assertIsInstance(result, EnrollmentResult)
        self.assertEqual(result.person_id, "alice_test")
        self.assertGreater(result.num_faces_processed, 0)

        # Test that person is in database
        enrolled_persons = face_rec.get_enrolled_persons()
        self.assertIn("alice_test", enrolled_persons)
        self.assertEqual(enrolled_persons["alice_test"]["name"], "Alice Test")

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_person_verification(self):
        """Test 1:1 person verification."""

        face_rec = FaceRecognition(hardware="cpu", db_path=self.db_path)

        # Enroll Alice first
        alice_images = list((self.test_images_dir / "alice").glob("*.jpg"))
        alice_images = [
            str(img) for img in alice_images[:2]
        ]  # Use first 2 for enrollment

        face_rec.enroll_person("alice_test", alice_images)

        # Test verification with enrolled person's image
        test_image = str(
            list((self.test_images_dir / "alice").glob("*.jpg"))[-1]
        )  # Use last image for test

        is_match, confidence = face_rec.verify_person(test_image, "alice_test")

        self.assertIsInstance(is_match, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test verification against non-existent person
        with self.assertRaises(ValueError):
            face_rec.verify_person(test_image, "non_existent_person")

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_person_identification(self):
        """Test 1:N person identification."""

        face_rec = FaceRecognition(hardware="cpu", db_path=self.db_path)

        # Enroll multiple people
        for person in ["alice", "bob"]:
            person_images = list((self.test_images_dir / person).glob("*.jpg"))
            person_images = [str(img) for img in person_images[:2]]

            face_rec.enroll_person(
                person_id=f"{person}_test",
                images=person_images,
                attributes={"name": f"{person.title()} Test"},
            )

        # Test identification
        test_image = str(list((self.test_images_dir / "alice").glob("*.jpg"))[-1])

        # Single match identification
        person_id, confidence = face_rec.identify_person(test_image)

        self.assertTrue(person_id is None or isinstance(person_id, str))
        self.assertIsInstance(confidence, float)

        # Top N identification
        candidates = face_rec.identify_person(test_image, return_top_n=3)

        self.assertIsInstance(candidates, list)
        for person_id, confidence in candidates:
            self.assertIsInstance(person_id, str)
            self.assertIsInstance(confidence, float)

    def test_enrollment_parameter_validation(self):
        """Test enrollment parameter validation."""

        # Skip actual model loading for this test
        # This would be mocked in a full unit test
        pass

    def test_database_management(self):
        """Test database management functions."""

        # Skip actual model loading for this test
        # This would test get_enrolled_persons, get_database_stats, etc.
        pass

    def test_face_quality_assessment(self):
        """Test face quality assessment functionality."""

        # This would test the _assess_face_quality method
        pass

    def test_error_handling(self):
        """Test error handling for various failure scenarios."""

        # Test handling of missing images
        # Test handling of images with no faces
        # Test handling of low-quality faces
        # Test database errors
        pass


class TestFaceQualityMetrics(unittest.TestCase):
    """Test cases for face quality metrics."""

    def test_face_quality_metrics_creation(self):
        """Test creation of FaceQualityMetrics."""

        metrics = FaceQualityMetrics(
            face_size=10000.0,
            confidence=0.95,
            landmark_quality=0.9,
            frontal_score=0.8,
            sharpness=0.85,
            brightness=0.75,
            overall_quality=0.82,
        )

        self.assertEqual(metrics.face_size, 10000.0)
        self.assertEqual(metrics.confidence, 0.95)
        self.assertEqual(metrics.overall_quality, 0.82)


class TestRecognitionResult(unittest.TestCase):
    """Test cases for recognition results."""

    def test_recognition_result_creation(self):
        """Test creation of RecognitionResult."""

        embedding = np.random.randn(512).astype(np.float32)
        quality = FaceQualityMetrics(
            face_size=10000.0,
            confidence=0.95,
            landmark_quality=0.9,
            frontal_score=0.8,
            sharpness=0.85,
            brightness=0.75,
            overall_quality=0.82,
        )

        result = RecognitionResult(
            person_id="test_person",
            confidence=0.85,
            embedding=embedding,
            quality=quality,
            bbox=[100, 100, 100, 100],
            landmarks=[np.array([120, 130]), np.array([180, 130])],
        )

        self.assertEqual(result.person_id, "test_person")
        self.assertEqual(result.confidence, 0.85)
        self.assertTrue(np.array_equal(result.embedding, embedding))
        self.assertEqual(result.quality, quality)


class TestEnrollmentResult(unittest.TestCase):
    """Test cases for enrollment results."""

    def test_enrollment_result_creation(self):
        """Test creation of EnrollmentResult."""

        result = EnrollmentResult(
            person_id="test_person",
            num_faces_processed=5,
            num_faces_enrolled=3,
            num_faces_rejected=2,
            quality_scores=[0.8, 0.7, 0.9],
            embedding_count=3,
        )

        self.assertEqual(result.person_id, "test_person")
        self.assertEqual(result.num_faces_processed, 5)
        self.assertEqual(result.num_faces_enrolled, 3)
        self.assertEqual(result.num_faces_rejected, 2)
        self.assertEqual(len(result.quality_scores), 3)
        self.assertEqual(result.embedding_count, 3)


class TestMultiFaceIdentification(unittest.TestCase):
    """Test cases for multi-face identification functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_faces.lance")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_image(self):
        """Create a mock image for testing."""
        return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_identify_all_persons_multi_face(self):
        """Test identifying all persons in a multi-face image."""

        face_rec = FaceRecognition(hardware="cpu", db_path=self.db_path)

        # This would test the identify_all_persons method with multiple faces
        # For now, this is a placeholder for integration testing
        pass

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_identify_all_persons_no_faces(self):
        """Test identifying all persons when no faces are detected."""

        face_rec = FaceRecognition(hardware="cpu", db_path=self.db_path)

        # Create image with no faces
        no_face_image = self.create_mock_image()

        results = face_rec.identify_all_persons(no_face_image)
        self.assertEqual(len(results), 0)

    @unittest.skip("Requires actual DeGirum models - enable for integration testing")
    def test_identify_all_persons_quality_rejection(self):
        """Test identifying all persons with quality-based rejection."""

        face_rec = FaceRecognition(hardware="cpu", db_path=self.db_path)

        # This would test quality-based rejection in multi-face scenarios
        # For now, this is a placeholder for integration testing
        pass


if __name__ == "__main__":
    # Configure test output
    unittest.main(verbosity=2)
