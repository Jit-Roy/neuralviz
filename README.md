# Unique NN Visualizer

A Python module for visualizing neural network training in real-time.

# Installation
You can install the unique_nn_visualizer package from PyPI using pip:
pip install unique-nn-visualizer

# Usage
Here’s a quick example of how to use unique_nn_visualizer:

from unique_nn_visualizer import NNVisualizer
visualizer = NNVisualizer(data_loader)
for epoch in range(num_epochs):
    # Train your model
    # ...
    
    # Update the visualization
    visualizer.plot_decision_boundary(epoch, loss.item())

# Features
Real-time Visualization: See how the decision boundary evolves during training.
Customizable: Easily integrate with your existing training loops.
Interactive Plots: Updates plots in real-time for a simulation-like experience.

# Requirements
Python 3.x
Matplotlib
PyTorch (if using neural network functionalities)

# Contributing
Feel free to contribute to the project! To contribute, please fork the repository and submit a pull request.