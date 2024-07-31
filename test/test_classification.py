# import torch
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# from neuralviz import ClassificationVisualizer
# import torch.nn as nn
# import torch.optim as optim

# X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = torch.FloatTensor(X)
# y = torch.LongTensor(y)

# visualizer = ClassificationVisualizer(
#     input_size=X.shape[1],
#     hidden_sizes=[10, 10],
#     output_size=len(set(y.numpy())), 
#     activation_functions=['ReLU', 'ReLU'],
#     criterion=nn.CrossEntropyLoss(),  
#     optimizer=optim.SGD,
# )

# visualizer.train(X, y, epochs=200, lr=0.01, batch_size=32)
# visualizer.visualize_feature_importance()
# visualizer.visualize_layer_contributions(X, y,layer_num=2)
# visualizer.visualize_activations(X)
# visualizer.visualize_gradient_flow()
# visualizer.visualize_activations(X,layer_num=0)




# import torch
# from sklearn.datasets import make_classification
# from sklearn.preprocessing import StandardScaler
# from neuralviz import ClassificationVisualizer
# import torch.nn as nn
# import torch.optim as optim

# # Generate binary classification data
# X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
#                            n_clusters_per_class=1, n_classes=2, random_state=42)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = torch.FloatTensor(X)
# y = torch.FloatTensor(y).view(-1, 1)  # Reshape y to (n_samples, 1)

# visualizer = ClassificationVisualizer(
#     input_size=X.shape[1],
#     hidden_sizes=[10, 10],
#     output_size=1,  # Change to 1 for binary classification
#     activation_functions=['ReLU', 'ReLU'],
#     criterion=nn.BCEWithLogitsLoss(),  # Use Binary Cross Entropy loss
#     optimizer=optim.SGD,
# )

# visualizer.train(X, y, epochs=200, lr=0.01, batch_size=32)
# visualizer.visualize_feature_importance()
# visualizer.visualize_layer_contributions(X, y, layer_num=2)
# visualizer.visualize_gradient_flow()
# visualizer.visualize_activations(X, layer_num=0)