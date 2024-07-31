from .base import NeuralNetworkVisualizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
import copy

class ClassificationVisualizer(NeuralNetworkVisualizer):
    def __init__(self, input_size, hidden_sizes, output_size, activation_functions=None, model=None,criterion=None, optimizer=None, figsize=(10, 8)):
        super().__init__(input_size, hidden_sizes, output_size, activation_functions, model,criterion, optimizer, figsize)

    def train(self, X, y, epochs=100, lr=0.01, batch_size=32, validation_split=0.2, early_stopping_patience=30):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)
        
        criterion = self.criterion if self.criterion else nn.CrossEntropyLoss()
        
        if self.optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            if isinstance(self.optimizer, type):
                optimizer = self.optimizer(self.model.parameters(), lr=lr)
            else:
                optimizer = self.optimizer
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
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
                train_acc = self.calculate_accuracy(X_train, y_train)
                val_acc = self.calculate_accuracy(X_val, y_val)
            
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_acc)
            self.history['val_metric'].append(val_acc)
            
            if epoch % 5 == 0:
                self.visualize_boundary(X, y, xx, yy, epoch, np.mean(train_losses))
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

    def calculate_accuracy(self, X, y):
        with torch.no_grad():
            outputs = self.model(X)
            if outputs.shape[1] > 1:  # Multiclass
                predicted = torch.argmax(outputs, dim=1)
            else:  # Binary
                predicted = (outputs > 0.5).float()
            return accuracy_score(y.numpy(), predicted.numpy())

    def visualize_boundary(self, X, y, xx, yy, epoch, loss, title=None):
        num_classes = len(torch.unique(y))
        if num_classes > 2:
            self.visualize_multiclass_boundary(X, y, xx, yy, epoch, loss, title)
        else:
            self.ax.clear()  # Clear the previous plot
            
            Z = self.model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
            if Z.shape[1] > 1:  # If the output has more than one dimension, it's using softmax
                Z = torch.argmax(Z, dim=1)
            else:
                Z = (Z > 0).float()  # For binary classification with BCEWithLogitsLoss
            Z = Z.detach().numpy().reshape(xx.shape)
            
            self.ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu_r)
            self.ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu_r, edgecolor='k')
            if title:
                self.ax.set_title(title)
            else:
                self.ax.set_title(f'Epoch {epoch}, Loss: {loss:.4f}')
            self.ax.set_xlabel('X1')
            self.ax.set_ylabel('X2')
            
            clear_output(wait=True)  # Clear previous output
            display(self.fig)

    def visualize_multiclass_boundary(self, X, y, xx, yy, epoch, loss, title=None):
        self.ax.clear()  # Clear the previous plot
        
        Z = self.model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = torch.argmax(Z, dim=1).detach().numpy().reshape(xx.shape)
        
        num_classes = len(torch.unique(y))
        cmap = plt.colormaps['rainbow'].resampled(num_classes)
        
        contour = self.ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap, levels=np.arange(num_classes + 1) - 0.5)
        
        # Ensure y is a 1D array
        y_plot = y.detach().numpy().flatten() if isinstance(y, torch.Tensor) else y.flatten()
        
        scatter = self.ax.scatter(X[:, 0], X[:, 1], c=y_plot, cmap=cmap, edgecolor='k')
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f'Epoch {epoch}, Loss: {loss:.4f}')
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        
        # Add a color bar if it doesn't exist
        if not hasattr(self, 'colorbar'):
            self.colorbar = plt.colorbar(scatter, label='Class', ticks=range(num_classes))
        else:
            self.colorbar.update_normal(scatter)
        
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
        plt.plot(self.history['train_metric'], label='Train Accuracy')
        plt.plot(self.history['val_metric'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        display(plt.gcf())
        plt.close()

    def visualize_feature_importance(self):
        weights = list(self.model.parameters())[0].detach().numpy()
        plt.figure(figsize=(10, 6))
        plt.imshow(weights, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('Feature Importance (Classification)')
        plt.xlabel('Input Features')
        plt.ylabel('Hidden Units')
        display(plt.gcf())
        plt.close()

    def visualize_neuron_contribution(self, X, y, layer_index, neuron_index):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        mesh_input = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original Decision Boundary
        self.plot_boundary(X, y, xx, yy, ax=axes[0], title="Original Decision Boundary")
        
        # Create a copy of the model with the specified neuron removed
        temp_model = self.remove_neuron(layer_index, neuron_index)
        original_model = self.model
        
        # Decision Boundary without the neuron
        self.model = temp_model
        self.plot_boundary(X, y, xx, yy, ax=axes[1], 
                                    title=f"Without Neuron {neuron_index} in Layer {layer_index}")
        
        # Calculate the difference in outputs
        with torch.no_grad():
            original_output = original_model(mesh_input).numpy()
            modified_output = temp_model(mesh_input).numpy()
        
        # Ensure the output is 2D (for binary classification)
        if original_output.ndim == 1:
            original_output = original_output.reshape(-1, 1)
            modified_output = modified_output.reshape(-1, 1)
        
        # Apply sigmoid and take the first column if it's multi-class
        original_probs = 1 / (1 + np.exp(-original_output[:, 0]))
        modified_probs = 1 / (1 + np.exp(-modified_output[:, 0]))
        
        difference = np.abs(original_probs - modified_probs).reshape(xx.shape)
        
        im = axes[2].imshow(difference, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='coolwarm')
        axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
        fig.colorbar(im, ax=axes[2], label='Absolute Difference in Output')
        axes[2].set_title(f"Contribution of Neuron {neuron_index}")
        axes[2].set_xlabel("X1")
        axes[2].set_ylabel("X2")
        
        plt.tight_layout()
        plt.show()

        # Restore the original model
        self.model = original_model

    def plot_boundary(self, X, y, xx, yy, ax=None, title=None):
        if ax is None:
            ax = plt.gca()
        
        num_classes = len(torch.unique(y))
        Z = self.model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = torch.argmax(Z, dim=1).detach().numpy().reshape(xx.shape)
        
        cmap = plt.colormaps['rainbow'].resampled(num_classes)
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')
        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        return scatter

    def visualize_layer_contributions(self, X, y, layer_num=None):
        layer_index = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                if layer_num is None or layer_index == layer_num:
                    num_neurons = layer.out_features
                    for i in range(num_neurons):
                        print(f"Visualizing contribution of neuron {i} in layer {layer_index}")
                        self.visualize_neuron_contribution(X, y, layer_index, i)
                        plt.close()
                if layer_num is not None and layer_index > layer_num:
                    break
                layer_index += 1