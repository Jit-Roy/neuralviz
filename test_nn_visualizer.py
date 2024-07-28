import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from nn_visualizer import NNVisualizer  # Import your custom module

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create synthetic data
def create_data():
    n_samples = 1000
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) > 0).astype(float)
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

# Prepare data loader
X, y = create_data()
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model
model = SimpleNN()

# Instantiate the visualizer
visualizer = NNVisualizer(model, data_loader)

# Training loop
num_epochs = 20
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Plot decision boundary using the visualizer
    visualizer.plot_decision_boundary(epoch, loss.item())
