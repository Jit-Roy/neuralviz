import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from neuralviz import NNVisualizer  # Import your custom module
from sklearn.datasets import make_moons

# Generate a sample dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).view(-1, 1)

# Create and train the model
visualizer = NNVisualizer(
    layer_sizes=[2, 10, 10, 1],
    activation_functions=['ReLU', 'ReLU']
)
visualizer.train(X, y, epochs=200, lr=0.01, batch_size=32)

# Visualize feature importance
visualizer.visualize_feature_importance()

# Visualize contribution of a specific neuron
visualizer.visualize_neuron_contribution(X, y, layer_index=0, neuron_index=5)

# Visualize contributions of all neurons in the first hidden layer
visualizer.visualize_layer_contributions(X, y, layer_index=0)

# Save and load the model
visualizer.save_model('my_model.pth')
visualizer.load_model('my_model.pth')
