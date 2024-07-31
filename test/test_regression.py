# import torch
# from sklearn.datasets import make_regression
# from sklearn.preprocessing import StandardScaler
# from neuralviz import RegressionVisualizer
# import torch.nn as nn
# import torch.optim as optim

# X, y = make_regression(n_samples=500, n_features=1, noise=100, random_state=42)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = torch.FloatTensor(X)
# y = torch.FloatTensor(y).view(-1, 1)

# visualizer = RegressionVisualizer(
#     input_size=X.shape[1],
#     hidden_sizes=[5, 5],
#     output_size=1,
#     activation_functions=['ReLU', 'ReLU'],
#     criterion=nn.MSELoss(),
#     optimizer=optim.Adam,
# )

# visualizer.train(X, y, epochs=200, lr=0.01, batch_size=32)
# visualizer.visualize_fit(X, y, xx=None, epoch=None, loss=None)
# visualizer.plot_learning_curves()
# visualizer.visualize_feature_importance()
# visualizer.visualize_neuron_contribution(X, y, layer_index=0, neuron_index=0)
# visualizer.visualize_residuals(X, y)
# visualizer.visualize_layer_contributions(X, y,layer_index=0)