from setuptools import setup, find_packages

setup(
    name='arterial_gnet',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        'numpy',
        'matplotlib',
        "pandas",
        "seaborn",
        "scikit-learn",
        "networkx",
        "mycolorpy"
    ],
)