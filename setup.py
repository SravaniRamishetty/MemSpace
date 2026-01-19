from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memspace",
    version="0.1.0",
    author="MemSpace Team",
    description="Hierarchical Scene Graphs for Persistent Robotic Memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "open-clip-torch>=2.20.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "rerun-sdk>=0.17.0",
        "open3d>=0.17.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "faiss-cpu>=1.7.4",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
