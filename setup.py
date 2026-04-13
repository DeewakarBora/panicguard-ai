"""
PanicGuard AI — Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="panicguard-ai",
    version="1.0.0",
    author="PanicGuard AI Team",
    description="AI-powered multi-agent system preventing panic-driven investment decisions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/panicguard-ai",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "panicguard-train=models.train_panic_model:main",
        ],
    },
)
