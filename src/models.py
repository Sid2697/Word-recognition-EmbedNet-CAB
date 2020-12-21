"""
This file containts various Linear Neural Networks
"""
import torch.nn as nn
import torch.nn.functional as F


class deep_network(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            hidden_layers_features,
            dropout=False):
        """
        in_features: type int, size of the input tensor to the neural network
        out_featurs: type int, size of the output tensor from neural network
        hidden_layers_features: type list, list of input size of the hidden
        layers example: [4098, 4098]
        dropout: type bool, if True dropout will be used for regularisation
        Referred from:
        https://stackoverflow.com/questions/58097924/how-to-create-variable-
        names-in-loop-for-layers-in-pytorch-neural-network
        """
        super(deep_network, self).__init__()
        assert type(hidden_layers_features) == list, "hidden_layers_features \
            is a list of input size of the hidden layers.\
                Example: [4098, 4098]"
        assert len(hidden_layers_features) != 0, "Please provide hidden \
            layer's information"
        self.dropout = dropout
        self.layers = nn.ModuleList()
        current_dim = in_features
        for hidden_dim in hidden_layers_features:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers.append(nn.Linear(current_dim, out_features))
        self.batch_norm = nn.BatchNorm1d(num_features=out_features)
        if self.dropout:
            self.drop = nn.Dropout()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout:
                x = self.drop(x)
        # L2 normalisation to match the normalisation of input embedding
        return F.normalize(self.layers[-1](x), p=2, dim=1)


class EmbedNet(nn.Module):
    """
    EmbedNet network class
    """
    def __init__(
            self,
            in_features,
            out_features,
            hidden_layers=[1024, 512, 256, 128]):
        super(EmbedNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        current_dim = self.in_features
        for hidden_dim in self.hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers.append(nn.Linear(current_dim, self.out_features))
        self.prelu = nn.PReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.prelu(layer(x))
        return F.normalize(x, p=2, dim=1)


class Siamese(nn.Module):
    """
    Siamese network class
    """
    def __init__(
            self,
            in_features,
            out_features,
            dropout=False,
            batch_norm=False):
        super(Siamese, self).__init__()
        self.batch_norm = batch_norm
        self.l1 = nn.Linear(in_features, 4096)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(4096)
            self.bn2 = nn.BatchNorm1d(4096)
        self.l2 = nn.Linear(4096, 4096)
        self.drop = nn.Dropout()
        self.dropout = dropout
        self.l3 = nn.Linear(4096, out_features)

    def forward(self, x):
        out = F.relu(self.l1(x), inplace=True)
        if self.batch_norm:
            out = self.bn1(out)
        if self.dropout:
            out = self.drop(out)
        out = F.relu(self.l2(out), inplace=True)
        if self.batch_norm:
            out = self.bn2(out)
        if self.dropout:
            out = self.drop(out)
        out = self.l3(out)
        return out
