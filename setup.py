from setuptools import setup

setup(
    name='yaocptool',
    version='0.3.1',
    packages=['yaocptool', 'yaocptool.mpc', 'yaocptool.util', 'yaocptool.methods', 'yaocptool.methods.base',
              'yaocptool.methods.classic', 'yaocptool.parallel', 'yaocptool.modelling', 'yaocptool.estimation',
              'yaocptool.stochastic', 'yaocptool.optimization'],
    url='https://github.com/marcoaaguiar/yaocptool',
    license='',
    author='marco',
    author_email='',
    description='',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'casadi',
        'sobol_seq'
    ]
)
