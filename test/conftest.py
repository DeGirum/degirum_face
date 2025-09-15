#
# conftest.py - DeGirum Face: pytest configuration file
# Copyright DeGirum Corp. 2025
#
# Contains common pytest configuration and common test fixtures
#
import sys, os
import tempfile
import pytest
from pathlib import Path

# add current directory to sys.path to debug tests locally without package installation
sys.path.insert(0, os.getcwd())


@pytest.fixture()
def temp_dir():
    """Create temporary directory for test databases and files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def assets_dir():
    """Get path to assets directory"""
    current_dir = Path(__file__).parent
    assets_dir = current_dir.parent / "examples" / "assets"
    if not assets_dir.exists():
        pytest.skip(f"Assets directory not found: {assets_dir}")
    return str(assets_dir)
