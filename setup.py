# setup.py

import os
import re
from setuptools import setup, find_packages

def read_version():
    """
    Reads the __version__ from mag_eq_nmr/__init__.py
    Assumes a line of the form: __version__ = "0.1.0"
    """
    version_file = os.path.join("mag_eq_nmr", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                # Extract everything after the '=' sign, remove quotes
                return re.sub(r'[^0-9a-zA-Z\.]', '', line.split("=")[1])
    return "0.0.0"  # fallback default

setup(
    name="MagEquivNMR",
    version=read_version(),
    description="Magnetic-Equivalence–Aware GNN for NMR Shift Prediction (AlgoMol)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AlgoMol",
    author_email="support@algomol.example",  # Replace with your actual email
    url="https://github.com/lanl2tz/AlgoMol",  # or your project repo
    packages=find_packages(),
    # If you want to install minimal dependencies automatically
    # you can either parse them from requirements.txt or hardcode them here:
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "rdkit-pypi",       # or 'rdkit' via conda
        "numpy>=1.20",
        "scipy>=1.7",
        "pandas>=1.0",
        "tqdm>=4.0"
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="nmr gnn deep-learning equivalence algomol",
)