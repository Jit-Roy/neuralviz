from setuptools import setup, find_packages

setup(
    name='unique_nn_visualizer',  # Ensure this is a unique name
    version='0.4',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib'
    ],
    # Other arguments
)