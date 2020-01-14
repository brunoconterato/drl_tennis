import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Adapted from Udacity codes
class Network(nn.Module):
    def __init__(self, input_dim, hidden_dimensions, output_dim, actor=False, allow_bn = True):
        super(Network, self).__init__()
        self.fc = []
        self.batch_norm = []
        self.allow_bn = allow_bn
        
        last_dimension = input_dim
        for i, dimension in enumerate(hidden_dimensions):
            self.fc.append(nn.Linear(last_dimension, dimension))
            self.add_module("fc"+str(i+1), self.fc[i])
            self.batch_norm.append(nn.BatchNorm1d(num_features=dimension))
            last_dimension = dimension
        self.fc.append(nn.Linear(last_dimension, output_dim))
        self.num_layers = len(self.fc)
        self.add_module("fc"+str(self.num_layers), self.fc[self.num_layers-1])
        
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers - 1):
            self.fc[i].weight.data.uniform_(*hidden_init(self.fc[i]))
        
        self.fc[-1].weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        skip_bn = False
        if x.shape[0] > 1:
            skip_bn = True
            
        if self.actor:
            for i in range(self.num_layers-1):
                x = self.fc[i](x)
                if self.allow_bn and not(skip_bn):
                    x = self.batch_norm[i](x)
                x = f.relu(x)
             
            x = self.fc[-1](x)
            norm = torch.norm(x)
            
            # x is a 2D vector (relative to horizontal movement and jump)
            # we bound the norm of the vector to be between 0 and 10
            return 10.0*(f.tanh(norm))*x/norm if norm > 0 else 10*x
        
        else:
            for i in range(self.num_layers-1):
                x = self.fc[i](x)
                if self.allow_bn and not(skip_bn):
                    x = self.batch_norm[i](x)
                x = f.relu(x)
            
            x = self.fc[-1](x)
            return x