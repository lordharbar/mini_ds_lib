from setuptools import setup, find_packages

setup(
    name="mini_ds_lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "lightgbm>=3.3.0",
        "catboost>=1.0.0",
        "pytorch>=2.0.0",
        "prophet>=1.1.0",
        "neuralprophet>=0.6.0",
    ],
    author="lordharbar",
    author_email="",
    description="Mini Data Science Workflow Library",
    keywords="data science, machine learning, automl",
    python_requires=">=3.9",
)