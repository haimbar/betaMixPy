import os
import re
import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the version from the package so it is defined in exactly one place.
with open(path.join(this_directory, 'betaMixPy', '__init__.py'), encoding='utf-8') as f:
    version = re.search(r'^__version__\s*=\s*"([^"]+)"', f.read(), re.M).group(1)

setuptools.setup(
    name="betaMixPy",
    version=version,
    author="Haim Bar",
    author_email="haim.bar@uconn.edu",
    description="A Python package to find strong correlations among P variables, each with N observations.",
    url = "https://pypi.org/project/betaMixPy/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['betaMixPy'],
    install_requires=["matplotlib", "pandas", "numpy", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
