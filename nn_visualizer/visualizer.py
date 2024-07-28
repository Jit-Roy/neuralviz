import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display, clear_output

class NNVisualizer:
    def __init__(self, model, data, targets):
        self.model = model
        self.data = data
        self.targets = targets
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title('Decision Boundary Evolution')
        self.xx, self.yy = self.create_meshgrid()

    def create_meshgrid(self):
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        return xx, yy

    def plot_decision_boundary(self, epoch, loss):
        self.ax.clear()  # Clear the previous plot

        # Compute the decision boundary
        Z = self.get_decision_boundary()
        
        # Plot decision boundary
        self.ax.contourf(self.xx, self.yy, Z, alpha=0.8, cmap='RdYlBu')
        
        # Plot data points
        self.ax.scatter(self.data[:, 0].numpy(), self.data[:, 1].numpy(), c=self.targets.squeeze().numpy(), cmap='coolwarm', edgecolor='k', s=20)
        
        # Set plot limits to ensure all data points are visible
        self.ax.set_xlim(self.xx.min(), self.xx.max())
        self.ax.set_ylim(self.yy.min(), self.yy.max())
        
        # Add labels and title
        self.ax.set_xlabel('Feature 1')
        self.ax.set_ylabel('Feature 2')
        self.ax.set_title(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
        
        # Update the plot
        clear_output(wait=True)
        display(self.figure)

    def get_decision_boundary(self):
        self.model.eval()
        with torch.no_grad():
            X_grid = torch.FloatTensor(np.c_[self.xx.ravel(), self.yy.ravel()])
            Z = self.model(X_grid).numpy()
        return Z.reshape(self.xx.shape)