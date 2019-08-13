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
    name="optim_methods",
    version="0.1",
    author="Cliff Kerr",
    author_email="cliff@thekerrlab.com",
    description="Optimization methods for IDM",
    keywords=["stochastic", "optimization"],
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