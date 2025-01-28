from setuptools import setup

__version__ = "0.0.27"
__author__ = "Rich Williams"
__author_email__ = (
    "rich@coralgardeners.org"
)
__url__ = "https://github.com/CoralGardeners/reefos-analysis.git"

setup(
    name="reefos-analysis",
    author=__author__,
    author_email=__author_email__,
    version=__version__,
    description="Shared analysis tools for various reefos data",
    python_requires=">=3.8",
    packages=["reefos_analysis", "reefos_analysis.dbutils"],
    include_package_data=True,
    zip_safe=True,
    url=__url__,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "soundfile",
        "influxdb_client",
        "firebase-admin",
        "ultralytics",
        "cachetools",
        "umap-learn",
        "hdbscan",
        "librosa"
    ],
)
