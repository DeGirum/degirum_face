from setuptools import setup, find_packages
from pathlib import Path

root_path = Path(__file__).resolve().parent

# get version
exec(open(root_path / "degirum_face/_version.py").read())


setup(
    name="degirum_face",
    version=__version__,  # noqa
    description="DeGirum Face Recognition Package",
    author="DeGirum Corp",
    license="DeGirum PySDK End-User License Agreement (https://docs.degirum.com/pysdk/eula)",
    author_email="support@degirum.com",
    url="https://github.com/DeGirum/degirum_face",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "degirum_tools[notifications] >= 0.21.0",
        "lancedb >= 0.24.0",
    ],
    extras_require={
        "build": ["twine", "build", "mypy", "flake8", "types-requests"],
    },
    include_package_data=True,
)
