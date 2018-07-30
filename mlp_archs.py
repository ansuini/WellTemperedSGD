import torch
import torch.nn as nn
import torch.nn.functional as F

# Multilayer Perceptron with variable number of units

class MLP(nn.Module):
    def __init__(self, h_sizes, out_size, actfun=F.relu):

        super(MLP, self).__init__()

        # Hidden layers  
        self.actfun = actfun
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            x = self.actfun(layer(x))
        
        output= F.log_softmax(self.out(x), dim=1)
        return output