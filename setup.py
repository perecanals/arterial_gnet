from setuptools import setup, find_packages

setup(
    name='arterial_net',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.9.0',
        'torch-geometric==1.7.2',
        'numpy==1.21.2',
        'matplotlib==3.4.3',
    ],
)