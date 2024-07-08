import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kan
from .utils import FLOPs_MAP

class KANbeFair(nn.Module):
    def __init__(self, args):
        super(KANbeFair, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            kan.KAN(
                width = [args.input_size] + args.layers_width + [args.output_size],
                grid = args.kan_bspline_grid,
                k = args.kan_bspline_order,
                symbolic_enabled=False,
                base_fun=args.kan_shortcut_function,
                grid_range = args.kan_grid_range
                )
            )

        self.layers_width = [args.input_size] + args.layers_width + [args.output_size]
        self.shortcut_function_name = args.kan_shortcut_name
        self.grid = args.kan_bspline_grid
        self.k = args.kan_bspline_order

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def print_kan(self):
        # for kan_model in self.layers:
        #     kan_model.print_kan()
        pass

    def layer_flops(self, din, dout, shortcut_name, grid, k):
        flops = (din * dout) * (9 * k * (grid +  1.5 * k) + 2 * grid - 2.5 * k + 1)
        if shortcut_name == "zero":
            shortcut_flops = 0
        else:
            shortcut_flops = FLOPs_MAP[shortcut_name] * din + 2 * din * dout
        return flops + shortcut_flops

    def layer_parameters(self, din, dout, shortcut_name, grid, k):
        parameters = din * dout * (grid + k + 2) + dout
        if shortcut_name == "zero":
            shortcut_parameters = 0
        else:
            shortcut_parameters = din * dout
        return parameters + shortcut_parameters

    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.shortcut_function_name, self.grid, self.k)
        return total_flops

    def total_parameters(self):
        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1], self.shortcut_function_name, self.grid, self.k)
        return total_parameters

class KANbeFair_Text(nn.Module):
    def __init__(self, args):
        super(KANbeFair_Text, self).__init__()
        self.embedding = nn.EmbeddingBag(args.input_size, args.layers_width[0], sparse=False)
        self.layers = nn.ModuleList()
        self.layers.append(
            kan.KAN(
                width = [args.layers_width[0]] + args.layers_width + [args.output_size],
                grid = args.kan_bspline_grid,
                k = args.kan_bspline_order,
                symbolic_enabled=False,
                base_fun=args.kan_shortcut_function,
                grid_range = args.kan_grid_range
                )
            )

        self.layers_width = [args.layers_width[0]] + args.layers_width + [args.output_size]
        self.shortcut_function_name = args.kan_shortcut_name
        self.grid = args.kan_bspline_grid
        self.k = args.kan_bspline_order

    def forward(self, inputs):
        text, offsets = inputs
        x = self.embedding(text, offsets)
        for layer in self.layers:
            x = layer(x)
        return x


    def print_kan(self):
        # for kan_model in self.layers:
        #     kan_model.print_kan()
        pass

    def layer_flops(self, din, dout, shortcut_name, grid, k):
        flops = (din * dout) * (9 * k * (grid +  1.5 * k) + 2 * grid - 2.5 * k + 1)
        if shortcut_name == "zero":
            shortcut_flops = 0
        else:
            shortcut_flops = FLOPs_MAP[shortcut_name] * din + 2 * din * dout
        return flops + shortcut_flops

    def layer_parameters(self, din, dout, shortcut_name, grid, k):
        parameters = din * dout * (grid + k + 2) + dout
        if shortcut_name == "zero":
            shortcut_parameters = 0
        else:
            shortcut_parameters = din * dout
        return parameters + shortcut_parameters

    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.shortcut_function_name, self.grid, self.k)
        return total_flops

    def total_parameters(self):
        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1], self.shortcut_function_name, self.grid, self.k)
        return total_parameters