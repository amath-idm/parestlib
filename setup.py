from setuptools import setup, find_packages

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GPLv3",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 1",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="parestlib",
    version="0.2",
    author="Cliff Kerr, Daniel Klein",
    author_email="ckerr@idmod.org",
    description="Parameter estimation library",
    keywords=["parameter", "estimation", "posterior", "sampling", "stochastic", "optimization"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib>=1.4.2",
        "numpy>=1.10.1",
        "scipy>=1.2.0",
        "sciris>=0.14.0",
    ],
)