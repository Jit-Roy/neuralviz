from .base import NeuralNetworkVisualizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
import copy

class RegressionVisualizer(NeuralNetworkVisualizer):
    def __init__(self, input_size, hidden_sizes, output_size, activation_functions=None, model=None,criterion=None, optimizer=None, figsize=(10, 8)):
        super().__init__(input_size, hidden_sizes, output_size, activation_functions, model,criterion, optimizer, figsize)

    def _clone_model(self):
        """ Create a deep copy of the model. """
        return copy.deepcopy(self.model)

    def _initialize_model(self):
        """ Initialize a new model with the same architecture. """
        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.output_size))
        return nn.Sequential(*layers)

    def get_model_with_removed_neuron(self, model, layer_index, neuron_index):
        """ Return a new model with the specified neuron removed. """
        new_model = self._clone_model()  # Clone the original model
        layers = list(new_model.children())
        
        # Access the specific layer
        layer = layers[layer_index]
        
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.clone()
            bias = layer.bias.data.clone()
            weight[neuron_index, :] = 0
            bias[neuron_index] = 0
            
            # Modify the weights and biases
            new_layer = nn.Linear(layer.in_features, layer.out_features)
            new_layer.weight.data = weight
            new_layer.bias.data = bias
            layers[layer_index] = new_layer

        new_model = nn.Sequential(*layers)
        return new_model

    def visualize_layer_contributions(self, X, y, layer_index):
        layers = list(self.model.children())
        if layer_index >= len(layers) or not isinstance(layers[layer_index], nn.Linear):
            raise ValueError(f"Layer at index {layer_index} is not a Linear layer")

        num_neurons = layers[layer_index].out_features
        for i in range(num_neurons):
            print(f"Visualizing contribution of neuron {i} in layer {layer_index}")
            
            # Model with one neuron removed
            model_without_neuron = self.get_model_with_removed_neuron(self.model, layer_index, i)

            # Original model predictions
            with torch.no_grad():
                original_output = self.model(X).detach().numpy()

            # Modified model predictions
            with torch.no_grad():
                modified_output = model_without_neuron(X).detach().numpy()

            # Compute difference
            difference = original_output - modified_output

            # Plotting
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            self.plot_fit(X, y, title="Original Fit")
            
            plt.subplot(1, 3, 2)
            self.plot_fit(X, y, title=f"Without Neuron {i} in Layer {layer_index}")

            plt.subplot(1, 3, 3)
            plt.scatter(X.numpy(), difference, c=difference.flatten(), cmap='coolwarm')
            plt.colorbar(label='Difference in Output')
            plt.title(f"Contribution of Neuron {i}")
            plt.xlabel("X")
            plt.ylabel("Difference")

            plt.tight_layout()
            plt.show()

    def train(self, X, y, epochs=100, lr=0.01, batch_size=32, validation_split=0.2, early_stopping_patience=30):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)
        
        criterion = nn.MSELoss()
        
        if self.optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            if isinstance(self.optimizer, type):
                optimizer = self.optimizer(self.model.parameters(), lr=lr)
            else:
                optimizer = self.optimizer
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        
        n_batches = len(X_train) // batch_size
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for i in range(n_batches):
                batch_X = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                train_r2 = self.calculate_r2(X_train, y_train)
                val_r2 = self.calculate_r2(X_val, y_val)
            
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_r2)
            self.history['val_metric'].append(val_r2)
            
            if epoch % 5 == 0:
                self.visualize_fit(X, y, xx, epoch, np.mean(train_losses))
                self.plot_learning_curves()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        plt.ioff()
        plt.show()

    def calculate_r2(self, X, y):
        with torch.no_grad():
            outputs = self.model(X)
            return r2_score(y.numpy(), outputs.numpy())

    def visualize_fit(self, X, y, xx=None, epoch=None, loss=None, title=None):
        self.ax.clear()  # Clear the previous plot

        if xx is None:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        
        xx_tensor = torch.FloatTensor(xx)
        y_pred = self.model(xx_tensor).detach().numpy()
        
        self.ax.scatter(X, y, color='blue', alpha=0.5, label='Data')
        self.ax.plot(xx, y_pred, color='red', label='Prediction')

        if title:
            self.ax.set_title(title)
        else:
            epoch_str = f'Epoch {epoch}' if epoch is not None else 'Epoch'
            loss_str = f'Loss: {loss:.4f}' if loss is not None else 'Loss'
            self.ax.set_title(f'{epoch_str}, {loss_str}')
            
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('y')
        self.ax.legend()
        
        clear_output(wait=True)  # Clear previous output
        display(self.fig)  

    def plot_learning_curves(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_metric'], label='Train R2')
        plt.plot(self.history['val_metric'], label='Validation R2')
        plt.xlabel('Epoch')
        plt.ylabel('R2 Score')
        plt.legend()
        
        plt.tight_layout()
        display(plt.gcf())
        plt.close()

    def visualize_feature_importance(self):
        weights = list(self.model.parameters())[0].detach().numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(range(weights.shape[1]), np.abs(weights).mean(axis=0))
        plt.title('Feature Importance (Regression)')
        plt.xlabel('Input Features')
        plt.ylabel('Average Absolute Weight')
        display(plt.gcf())
        plt.close()

    def visualize_neuron_contribution(self, X, y, layer_index, neuron_index):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        self.plot_fit(X, y, title="Original Fit")
        
        temp_model = self.remove_neuron(layer_index, neuron_index)
        original_model = self.model
        self.model = temp_model
        plt.subplot(1, 3, 2)
        self.plot_fit(X, y, title=f"Without Neuron {neuron_index} in Layer {layer_index}")
        
        original_output = original_model(X).detach().numpy()
        modified_output = temp_model(X).detach().numpy()
        difference = original_output - modified_output
        
        plt.subplot(1, 3, 3)
        plt.scatter(X, difference, c=difference.flatten(), cmap='coolwarm')
        plt.colorbar(label='Difference in Output')
        plt.title(f"Contribution of Neuron {neuron_index}")
        plt.xlabel("X")
        plt.ylabel("Difference")
        
        plt.tight_layout()
        plt.show()

        self.model = original_model

    def plot_fit(self, X, y, title=None):
        # Sort X and y
        sorted_indices = X.numpy().argsort(axis=0).flatten()
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # No need to convert to tensor, as X_sorted is already a tensor
        y_pred = self.model(X_sorted).detach().numpy()
        
        plt.scatter(X.numpy(), y.numpy(), color='blue', alpha=0.5, label='Data')
        plt.plot(X_sorted.numpy(), y_pred, color='red', label='Prediction')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

    def visualize_residuals(self, X, y):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).numpy()
        
        residuals = y.numpy() - y_pred
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        plt.tight_layout()
        display(plt.gcf())
        plt.close()