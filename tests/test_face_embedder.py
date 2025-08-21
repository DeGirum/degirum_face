#!/usr/bin/env python3
"""
Test: Face Embedding Module

Basic tests for the FaceEmbedder class to ensure proper functionality.

Copyright DeGirum Corporation 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from degirum_face.face_embedder import FaceEmbedder, embed_face, verify_faces


class TestFaceEmbedder:
    """Test suite for FaceEmbedder class"""

    def test_face_embedder_init_requires_hardware(self):
        """Test that FaceEmbedder requires hardware parameter"""
        with pytest.raises(ValueError, match="Hardware must be specified"):
            FaceEmbedder()

    @patch("degirum_face.face_embedder.get_model_config")
    @patch("degirum_face.face_embedder.dg.connect")
    def test_face_embedder_auto_mode(self, mock_connect, mock_config):
        """Test FaceEmbedder auto mode initialization"""
        # Mock configuration
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.validate_hardware_task_combination.return_value = True
        mock_config_instance.get_zoo_url.return_value = "degirum/hailo"
        mock_config_instance.get_default_model.return_value = "arcface_mobilefacenet"

        # Mock DeGirum connection
        mock_zoo = Mock()
        mock_model = Mock()
        mock_connect.return_value = mock_zoo
        mock_zoo.load_model.return_value = mock_model

        # Create embedder
        embedder = FaceEmbedder("hailo8")

        # Verify initialization
        assert embedder.hardware == "hailo8"
        assert embedder.model == mock_model
        assert embedder.model_name == "arcface_mobilefacenet"
        assert embedder.TASK == "face_embedding"

    @patch("degirum_face.face_embedder.dg.connect")
    def test_face_embedder_custom(self, mock_connect):
        """Test FaceEmbedder custom mode"""
        # Mock DeGirum connection
        mock_zoo = Mock()
        mock_model = Mock()
        mock_connect.return_value = mock_zoo
        mock_zoo.load_model.return_value = mock_model

        # Create embedder with custom model
        embedder = FaceEmbedder.custom(model_name="custom_model", zoo_url="custom/zoo")

        # Verify initialization
        assert embedder.model == mock_model
        assert embedder.model_name == "custom_model"
        assert embedder.zoo_url == "custom/zoo"

    def test_get_model_info(self):
        """Test model information retrieval"""
        mock_model = Mock()
        embedder = FaceEmbedder(
            hardware="hailo8",
            model=mock_model,
            model_name="test_model",
            zoo_url="test/zoo",
            inference_host_address="@cloud",
        )

        info = embedder.get_model_info()

        assert info["model_name"] == "test_model"
        assert info["hardware"] == "hailo8"
        assert info["task"] == "face_embedding"
        assert info["creation_mode"] == "auto"

    @patch("degirum_face.face_embedder.FaceEmbedder.auto")
    def test_convenience_embed_face(self, mock_auto):
        """Test convenience function for single embedding"""
        mock_embedder = Mock()
        mock_result = Mock()
        mock_embedder.embed.return_value = mock_result
        mock_auto.return_value = mock_embedder

        result = embed_face("test.jpg", hardware="hailo8")

        assert result == mock_result
        mock_auto.assert_called_once_with(
            hardware="hailo8", inference_host_address="@cloud"
        )
        mock_embedder.embed.assert_called_once_with("test.jpg")

    def test_repr(self):
        """Test string representation"""
        mock_model = Mock()

        # Test auto mode
        embedder = FaceEmbedder(
            hardware="hailo8", model=mock_model, model_name="test_model"
        )
        assert repr(embedder) == "FaceEmbedder[auto](model='test_model')"

        # Test custom mode
        embedder = FaceEmbedder(model=mock_model, model_name="test_model")
        assert repr(embedder) == "FaceEmbedder[custom](model='test_model')"


if __name__ == "__main__":
    pytest.main([__file__])
