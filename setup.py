from setuptools import setup, find_packages

setup(
    name='nn_visualizer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        # Add other dependencies here
    ],
)
