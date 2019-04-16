from setuptools import setup, find_packages

import metadata

setup(
    name='yaocptool',
    version=metadata.__version__,
    packages=find_packages(where='yaocptool'),
    package_dir={'': 'yaocptoolpyh'},
    url='https://github.com/marcoaaguiar/yaocptool',
    license='',
    author=metadata.__author__,
    author_email=metadata.__authoremail__,
    description='',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'casadi',
        'sobol_seq',
        'networkx'
    ]
)
