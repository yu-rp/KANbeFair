import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FLOPs_MAP
from .kan.spline import *

class BSpline(nn.Module):
    def __init__(self, input_dim, grid_num, grid_range, k):
        super(BSpline, self).__init__()
        grid = torch.einsum('i,j->ij', torch.ones(input_dim), torch.linspace(grid_range[0], grid_range[1], steps=grid_num + 1))
        self.register_buffer('grid', grid)

        noises = (torch.rand_like(grid) - 1 / 2) * 0.1 / grid_num
        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k))
        self.k = k

    def forward(self, x):
        assert x.shape[1] == self.grid.shape[0]
        x = x.T
        x = coef2curve(
            x_eval=x, 
            grid=self.grid, 
            coef=self.coef, 
            k=self.k, 
            device=x.device)
        x = x.T
        return x


class BSpline_MLP(nn.Module):
    def __init__(self, args):
        super(BSpline_MLP, self).__init__()
        layers_width = [args.input_size] + args.layers_width
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_width) - 1):
            self.layers.append(nn.Linear(layers_width[i], layers_width[i+1]))
            if args.batch_norm:
                self.layers.append(nn.BatchNorm1d(layers_width[i+1]))
            self.layers.append(
                BSpline(
                    input_dim=layers_width[i+1], 
                    grid_num=args.kan_bspline_grid, 
                    grid_range=args.kan_grid_range, 
                    k=args.kan_bspline_order
                )
            )
        self.layers.append(nn.Linear(args.layers_width[-1], args.output_size))

        self.layers_width = layers_width + [args.output_size] 
        self.grid_num = args.kan_bspline_grid
        self.k = args.kan_bspline_order
        self.grid_range = args.kan_grid_range

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def layer_flops(self, din, dout, k, grid_num):
        return 2 * din * dout + dout * (9 * k * (grid_num + 1.5 * k) + 2 * grid_num - 2.5 * k - 1)

    def layer_parameters(self, din, dout, k, grid_num):
        return dout * (din + 1) + dout * (grid_num + k)

    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.k, self.grid_num)
        return total_flops

    def total_parameters(self):
        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1], self.k, self.grid_num)
        return total_parameters

class BSpline_First_MLP(nn.Module):
    def __init__(self, args):
        super(BSpline_First_MLP, self).__init__()
        layers_width = [args.input_size] + args.layers_width + [args.output_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_width) - 1):
            self.layers.append(
                BSpline(
                    input_dim=layers_width[i], 
                    grid_num=args.kan_bspline_grid, 
                    grid_range=args.kan_grid_range, 
                    k=args.kan_bspline_order
                )
            )
            self.layers.append(nn.Linear(layers_width[i], layers_width[i+1]))
            if args.batch_norm:
                self.layers.append(nn.BatchNorm1d(layers_width[i+1]))

        self.layers_width = layers_width
        self.grid_num = args.kan_bspline_grid
        self.k = args.kan_bspline_order
        self.grid_range = args.kan_grid_range
        print(self)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def layer_flops(self, din, dout, k, grid_num):
        return 2 * din * dout + din * (9 * k * (grid_num + 1.5 * k) + 2 * grid_num - 2.5 * k - 1)

    def layer_parameters(self, din, dout, k, grid_num):
        return dout * (din + 1) + din * (grid_num + k)

    def total_flops(self):
        total_flops = 0
        for i in range(len(self.layers_width) - 1):
            total_flops += self.layer_flops(self.layers_width[i], self.layers_width[i+1], self.k, self.grid_num)
        return total_flops

    def total_parameters(self):
        total_parameters = 0
        for i in range(len(self.layers_width) - 1):
            total_parameters += self.layer_parameters(self.layers_width[i], self.layers_width[i+1], self.k, self.grid_num)
        return total_parameters