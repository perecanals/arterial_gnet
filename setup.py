from setuptools import setup, find_packages

setup(
    name='arterial_net',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        'numpy',
        'matplotlib',
    ],
)