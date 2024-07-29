import torch
from neuralviz import NNVisualizer  
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).view(-1, 1)

visualizer = NNVisualizer(
    layer_sizes=[2, 10, 10, 1],
    activation_functions=['ReLU', 'ReLU']
)

visualizer.train(X, y, epochs=200, lr=0.01, batch_size=32)
visualizer.visualize_feature_importance()
visualizer.visualize_neuron_contribution(X, y, layer_index=0, neuron_index=5)
visualizer.visualize_layer_contributions(X, y, layer_index=0)