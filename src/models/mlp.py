import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FLOPs_MAP

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        layers_width = [args.input_size] + args.layers_width
        self.layers = nn.ModuleList()
        for i in range(len(layers_width) - 1):
            self.layers.append(nn.Linear(layers_width[i], layers_width[i+1]))
            if args.batch_norm:
                self.layers.append(nn.BatchNorm1d(layers_width[i+1]))
            self.layers.append(args.activation())
        self.layers.append(nn.Linear(args.layers_width[-1], args.output_size))

        self.layers_width = layers_width + [args.output_size] 
        self.activation_name = args.activation_name
        self.batch_norm = args.batch_norm

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def layer_flops(self, din, dout, nonlinearity):
        return 2 * din * dout + FLOPs_MAP[nonlinearity] * dout

    def layer_parameters(self, din, dout):
        return dout * (din + 1)

    def batchnorm_flops(self, dout):
        return dout * 4

    def batchnorm_parameters(self, dout):
        return 2 * dout

    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.activation_name)
            if self.batch_norm:
                total_flops += self.batchnorm_flops(self.layers_width[i+1])
        return total_flops

    def total_parameters(self):
        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1])
            if self.batch_norm:
                total_parameters += self.batchnorm_parameters(self.layers_width[i+1])
        return total_parameters

class MLP_Text(nn.Module):
    def __init__(self, args):
        super(MLP_Text, self).__init__()
        self.embedding = nn.EmbeddingBag(args.input_size, args.layers_width[0], sparse=False)
        layers_width = [args.layers_width[0]] + args.layers_width
        self.layers = nn.ModuleList()
        for i in range(len(layers_width) - 1):
            self.layers.append(nn.Linear(layers_width[i], layers_width[i+1]))
            if args.batch_norm:
                self.layers.append(nn.BatchNorm1d(layers_width[i+1]))
            self.layers.append(args.activation())
        self.layers.append(nn.Linear(args.layers_width[-1], args.output_size))

        self.layers_width = layers_width + [args.output_size] 
        self.activation_name = args.activation_name
        self.batch_norm = args.batch_norm

    def forward(self, inputs):
        text, offsets = inputs
        x = self.embedding(text, offsets)
        for layer in self.layers:
            x = layer(x)
        return x

    def layer_flops(self, din, dout, nonlinearity):
        return 2 * din * dout + FLOPs_MAP[nonlinearity] * dout

    def layer_parameters(self, din, dout):
        return dout * (din + 1)

    def batchnorm_flops(self, dout):
        return dout * 4

    def batchnorm_parameters(self, dout):
        return 2 * dout
    
    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.activation_name)
            if self.batch_norm:
                total_flops += self.batchnorm_flops(self.layers_width[i+1])
        return total_flops

    def total_parameters(self):
        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1])
            if self.batch_norm:
                total_parameters += self.batchnorm_parameters(self.layers_width[i+1])
        return total_parameters