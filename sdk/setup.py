from setuptools import setup, find_packages

setup(
    name="agentfences",
    version="0.1.0",
    description="Runtime governance for AI agents — budget limits, loop protection, token limits, audit trail",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bhavik Sardar",
    url="https://github.com/bhaviksardar/fences",
    packages=find_packages(include=["fences", "fences.*"]),
    install_requires=[
        "requests>=2.25.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["ai", "agents", "governance", "llm", "budget", "safety", "token"],
)