import matplotlib.pyplot as plt
import torch.nn as nn
from IPython.display import display, clear_output
import copy
import seaborn as sns

class NeuralNetworkVisualizer:
    def __init__(self, input_size, hidden_sizes, output_size, activation_functions=None, model=None,criterion=None, optimizer=None, figsize=(10, 8)):
            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.output_size = output_size
            self.activation_functions = activation_functions
            self.optimizer = optimizer
            self.criterion = criterion
            self.fig, self.ax = plt.subplots(figsize=figsize)
            display(self.fig)
            self.history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
            if model is None:
                self.model = self.create_model()
            else:
                self.model = model

    def get_activation(self, activation):
        if isinstance(activation, str):
            return getattr(nn, activation)()
        elif callable(activation):
            return nn.Module(activation)
        else:
            raise ValueError(f"Unsupported activation type: {type(activation)}")

    def create_model(self):
        if self.activation_functions is None:
            raise ValueError("activation_functions must be provided if no custom model is given.")
        
        class Lambda(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  
                activation = self.activation_functions[i]
                if isinstance(activation, str):
                    layers.append(getattr(nn, activation)())
                elif callable(activation):
                    layers.append(Lambda(activation))
                else:
                    raise ValueError(f"Unsupported activation type: {type(activation)}")
        return nn.Sequential(*layers)

    def remove_neuron(self, layer_index, neuron_index):
        temp_model = copy.deepcopy(self.model)
        layers = list(temp_model.children())
        
        if isinstance(layers[layer_index], nn.Linear):
            # Zero out the weights and bias for the specified neuron
            layers[layer_index].weight.data[neuron_index, :] = 0
            layers[layer_index].bias.data[neuron_index] = 0
            
            # If this isn't the last layer, we also need to zero out the connections to the next layer
            if layer_index < len(layers) - 1 and isinstance(layers[layer_index + 1], nn.Linear):
                layers[layer_index + 1].weight.data[:, neuron_index] = 0
        
        return temp_model

    def visualize_activations(self, X, layer_num=None):
        activations = []
        self.model.eval()
        x = X.clone()
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if layer_num is None or i == layer_num:
                    activations.append((i, x.detach().numpy()))
            else:
                x = layer(x)
            
            if layer_num is not None and i > layer_num:
                break
        
        num_plots = len(activations)
        if num_plots == 0:
            print("No activations to visualize.")
            return
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        for ax, (layer_index, activation) in zip(axes, activations):
            sns.heatmap(activation.T, ax=ax, cmap='viridis')
            ax.set_title(f'Layer {layer_index}')
        
        plt.tight_layout()
        plt.show()

    def visualize_gradient_flow(self):
        grad_flow = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_flow.append((name, param.grad.abs().mean().item()))
        
        names, grads = zip(*grad_flow)
        plt.figure(figsize=(10, 5))
        plt.bar(names, grads)
        plt.xticks(rotation=90)
        plt.xlabel('Layers')
        plt.ylabel('Average Gradient')
        plt.title(f'Gradient Flow')
        plt.tight_layout()
        plt.show()