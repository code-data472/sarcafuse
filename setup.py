from setuptools import setup, find_packages

setup(
    name="sarcafuse",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "torch-geometric>=2.4.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "openai>=1.3.0",
    ],
)
