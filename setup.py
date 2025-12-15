from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pacman-rl",
    version="0.1.0",
    author="SABRYOLA",
    description="Reinforcement Learning for Ms. Pac-Man using DQN, PPO, and A2C",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SABRYOLA/pacman",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium[atari]>=0.29.0",
        "ale-py>=0.8.1",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "tensorboard>=2.14.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
