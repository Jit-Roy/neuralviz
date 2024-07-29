import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy

class NeuralNetworkVisualizer:
    def __init__(self, model=None, activation_functions=None, layer_sizes=None, criterion=None, optimizer=None, figsize=(10, 8)):
        self.model = model
        self.activation_functions = activation_functions
        self.layer_sizes = layer_sizes
        self.criterion = criterion
        self.optimizer = optimizer
        if model is None:
            self.model = self.create_model()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        display(self.fig)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def create_model(self):
        if self.layer_sizes is None or self.activation_functions is None:
            raise ValueError("layer_sizes and activation_functions must be provided if no custom model is given.")
        
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            if i < len(self.layer_sizes) - 2:  
                layers.append(self.get_activation(self.activation_functions[i]))
        return nn.Sequential(*layers)

    def get_activation(self, name):
        activation_function = getattr(nn, name, None)
        if activation_function is None:
            raise ValueError(f"Activation function {name} is not recognized.")
        return activation_function()

    def train(self, X, y, epochs=100, lr=0.01, batch_size=32, validation_split=0.2, early_stopping_patience=10):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)
        
        criterion = self.criterion or nn.BCEWithLogitsLoss()
        optimizer = self.optimizer(self.model.parameters(), lr=lr) if self.optimizer else optim.Adam(self.model.parameters(), lr=lr)
        
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
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if epoch % 5 == 0:
                self.visualize_decision_boundary(X, y, xx, yy, epoch, np.mean(train_losses))
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
            predicted = (outputs > 0.5).float()
            return accuracy_score(y.numpy(), predicted.numpy())

    def visualize_decision_boundary(self, X, y, xx, yy, epoch, loss, title=None):
        self.ax.clear()  # Clear the previous plot
        
        Z = self.model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.detach().numpy().reshape(xx.shape)
        
        self.ax.contourf(xx, yy, Z > 0, alpha=0.8, cmap=plt.cm.RdYlBu_r)
        self.ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu_r, edgecolor='k')
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f'Epoch {epoch}, Loss: {loss:.4f}')
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        
        clear_output(wait=True)  # Clear previous output
        display(self.fig)  # Redisplay updated figure

    def plot_learning_curves(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
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
        plt.title('Feature Importance')
        plt.xlabel('Input Features')
        plt.ylabel('Hidden Units')
        display(plt.gcf())
        plt.close()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")

    def remove_neuron(self, layer_index, neuron_index):
        temp_model = copy.deepcopy(self.model)
        layer = list(temp_model.children())[layer_index]
        
        if isinstance(layer, nn.Linear):
            layer.weight.data[neuron_index, :] = 0
            layer.bias.data[neuron_index] = 0
        
        return temp_model

    def visualize_neuron_contribution(self, X, y, layer_index, neuron_index):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))

        # Original decision boundary
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        self.plot_decision_boundary(X, y, xx, yy, title="Original Decision Boundary")
        
        # Decision boundary without the specified neuron
        temp_model = self.remove_neuron(layer_index, neuron_index)
        original_model = self.model
        self.model = temp_model
        plt.subplot(1, 3, 2)
        self.plot_decision_boundary(X, y, xx, yy, 
                                    title=f"Without Neuron {neuron_index} in Layer {layer_index}")
        
        # Calculate the difference
        original_output = original_model(torch.FloatTensor(X)).detach().numpy()
        modified_output = temp_model(torch.FloatTensor(X)).detach().numpy()
        difference = original_output - modified_output
        
        # Visualize the difference
        plt.subplot(1, 3, 3)
        plt.scatter(X[:, 0], X[:, 1], c=difference.flatten(), cmap='coolwarm')
        plt.colorbar(label='Difference in Output')
        plt.title(f"Contribution of Neuron {neuron_index}")
        plt.xlabel("X1")
        plt.ylabel("X2")
        
        plt.tight_layout()
        plt.show()

        # Restore the original model
        self.model = original_model

    def plot_decision_boundary(self, X, y, xx, yy, title=None):
        Z = self.model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.detach().numpy().reshape(xx.shape)
        
        plt.contourf(xx, yy, Z > 0, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')

    def visualize_layer_contributions(self, X, y, layer_index):
        layer = list(self.model.children())[layer_index]
        if isinstance(layer, nn.Linear):
            num_neurons = layer.out_features
            for i in range(num_neurons):
                print(f"Visualizing contribution of neuron {i} in layer {layer_index}")
                self.visualize_neuron_contribution(X, y, layer_index, i)