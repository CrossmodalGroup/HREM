import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bn=True, ln=False):
        super().__init__()

        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.bn = bn
        self.ln = ln

        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if self.bn:
            self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):

        if len(x.size()) == 3 and self.bn:
            B, N, D = x.size()
            # The data size has been changed for the compatibility of BatchNorm Layer,
            # The original data shape is B*N*D, 
            # while the bn layer needs the data whose shape is B*D
            x = x.reshape(B*N, D)
        else:
            N = 0

        if self.bn:
            for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)        
        
        if N > 0:
            x = x.view(B, N, self.output_dim)

        return x


class FC_MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bn=True):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, bn)
    
    def forward(self, x):

        x = self.fc(x) + self.mlp(x)
        return x


if __name__ == '__main__':
    
    pass
