"""Package setup for MedSeg Pipeline.

Install in editable mode with:
    pip install -e .

Or as a regular install:
    pip install .
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the long description from README
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

# Read dependencies from requirements.txt (exclude comments and blank lines)
requirements = []
req_path = this_dir / "requirements.txt"
if req_path.exists():
    with open(req_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)

setup(
    name="medseg-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@jhu.edu",
    description=(
        "Production-quality deep learning pipeline for medical image segmentation "
        "with U-Net and Attention U-Net models."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medseg-pipeline",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.2.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupytext>=1.14.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medseg-train=scripts.train:main",
            "medseg-eval=scripts.evaluate:main",
            "medseg-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "docs/*.md"],
    },
    keywords=[
        "medical imaging",
        "image segmentation",
        "deep learning",
        "brain tumor",
        "U-Net",
        "BraTS",
        "DICOM",
        "NIfTI",
        "PyTorch",
        "FDA GMLP",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/medseg-pipeline/issues",
        "Source":      "https://github.com/yourusername/medseg-pipeline",
        "Documentation": "https://github.com/yourusername/medseg-pipeline#readme",
    },
)
