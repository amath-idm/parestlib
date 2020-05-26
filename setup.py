import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'parestlib', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.md'), "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="parestlib",
    version=version,
    author="Cliff Kerr, Daniel Klein",
    author_email="covasim@idmod.org",
    description="Parameter estimation library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://covasim.org',
    keywords=["parameter", "estimation", "posterior", "sampling", "stochastic", "optimization"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib>=2.2.2",
        "numpy>=1.10.1",
        "scipy>=1.2.0",
        "sciris>=0.15.6",
        "statsmodels",
    ],
)
