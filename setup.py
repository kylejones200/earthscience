from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="earthsciences",
    version="0.3.0",
    author="Earth Sciences Python Project",
    description="Python library for earth sciences data analysis (ALPHA VERSION)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "PyWavelets>=1.4.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
    ],
    entry_points={
        'console_scripts': [
            'earthsciences=earthsciences.cli:cli',
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=6.2.1",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=7.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.25.0",
        ],
        "geospatial": [
            "rasterio>=1.3.0",
            "geopandas>=0.13.0",
        ],
        "full": [
            "seaborn>=0.12.0",
            "Pillow>=10.0.0",
            "astropy>=5.3.0",
            "spectrum>=0.8.1",
        ],
    },
)
