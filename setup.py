"""Setup script for ColabLink."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="colablink",
    version="1.0.0",
    description="Connect your local IDE to Google Colab GPU runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ColabLink Contributors",
    license="MIT",
    url="https://github.com/PoshSylvester/colablink",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "pyngrok>=5.0.0",
    ],
    entry_points={
        "console_scripts": [
            "colablink=colablink.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="colab jupyter gpu remote development vscode cursor link",
)

