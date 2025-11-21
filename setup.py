"""Setup script for faster-gigaam."""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
version = {}
with open(os.path.join("faster_gigaam", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="faster-gigaam",
    version=version.get("__version__", "0.1.0"),
    author="Your Name",
    author_email="your.email@example.com",
    description="Optimized inference engine for GigaAM ASR models with CUDA acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/faster-gigaam",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Note: PyTorch is NOT included here - users should install it separately
        # to choose between CPU and CUDA versions
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.82.0",
        ],
    },
)
