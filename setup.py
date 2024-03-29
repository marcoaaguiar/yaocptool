from setuptools import find_packages, setup

import metadata

setup(
    name="yaocptool",
    version=metadata.__version__,
    packages=find_packages(include=["yaocptool*"]),
    url="https://github.com/marcoaaguiar/yaocptool",
    license="",
    author=metadata.__author__,
    author_email=metadata.__authoremail__,
    description="",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "casadi",
        "sobol_seq",
        "networkx",
    ],
)
