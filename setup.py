from setuptools import setup

from yaocptool import metadata


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='yaocptool',
    version=metadata.__version__,
    packages=['yaocptool', 'yaocptool.mpc', 'yaocptool.util', 'yaocptool.methods', 'yaocptool.methods.base',
              'yaocptool.methods.classic', 'yaocptool.parallel', 'yaocptool.modelling', 'yaocptool.estimation',
              'yaocptool.stochastic', 'yaocptool.optimization'],
    url='https://github.com/marcoaaguiar/yaocptool',
    license='',
    author=metadata.__author__,
    author_email=metadata.__authoremail__,
    description='',
    long_description=readme(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'casadi',
        'sobol_seq',
        'networkx'
    ]
)
