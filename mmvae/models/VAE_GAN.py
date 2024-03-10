import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=3):
        """
        Initializes the Discriminator model.
        :param input_size: Size of the latent representation from the VAE encoder.
        :param hidden_sizes: List containing the sizes of the hidden layers.
        :param output_size: Size of the output layer, corresponding to the number of species, default is 3.
        """
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        
        # From input layer to the first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Adding more hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Ensure the model is compatible with CUDA
        self.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def forward(self, x):
        """
        Forward pass of the discriminator.
        :param x: The latent representation from the VAE encoder.
        """
        for layer in self.layers:
            x = F.relu(layer(x)) # Using ReLU activation function for hidden layers
        x = self.output_layer(x) # Output layer without activation
        return F.softmax(x, dim=-1) # Applying softmax to output for probability distribution
