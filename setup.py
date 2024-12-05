from setuptools import setup, find_packages

setup(
    name="ml_pipeline_tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["scikit-learn", "pandas", "numpy", "joblib"],
    entry_points={"console_scripts": ["ml-pipeline=ml_pipeline_tool.main:main"]},
)