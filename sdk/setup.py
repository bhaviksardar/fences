from setuptools import setup, find_packages

setup(
    name="fences",
    version="0.1.0",
    description="Agent governance and runtime policy enforcement SDK",
    packages=find_packages(include=["fences", "fences.*"]),
    install_requires=[
        "requests>=2.25.0",
    ],
    python_requires=">=3.9",
)