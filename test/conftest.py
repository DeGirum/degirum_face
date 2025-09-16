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


def pytest_addoption(parser):
    """Add command line parameters"""
    parser.addoption(
        "--token", action="store", default="", help="cloud server token value to use"
    )


@pytest.fixture(scope="session", autouse=True)
def cloud_token(request):
    """Get cloud server token passed from the command line and install it system-wide"""
    token = request.config.getoption("--token")
    if token:
        from degirum._tokens import TokenManager

        TokenManager().token_install(token, True)


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
