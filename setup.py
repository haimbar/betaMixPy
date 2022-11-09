import os
import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setuptools.setup(
    name="betaMixPy",
    version="0.1.2",
    author="Haim Bar",
    author_email="haim.bar@uconn.edu",
    description="A Python package to find strong correlations among P variables, each with N observations.",
    url = "https://pypi.org/project/betaMixPy/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['betaMixPy'],
    install_requires=["statistics", "matplotlib", "pandas", "networkx", "numpy", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
