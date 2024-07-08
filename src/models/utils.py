FLOPs_MAP = {
    "zero": 0,
    "identity": 0,
    "relu": 1,
    'square_relu': 2,
    "sigmoid":4,
    "silu":5,
    "tanh":6,
    "gelu": 14,
    "polynomial2": 1+2+3-1,
    "polynomial3": 1+2+3+4-1,
    "polynomial5": 1+2+3+4+5-1,
}

import torch
import torch.nn as nn

class Square_ReLU(nn.Module):
    def __init__(self):
        super(Square_ReLU, self).__init__()

    def forward(self, x):
        return torch.relu(x)**2

class Polynomial2(nn.Module):
    def __init__(self):
        super(Polynomial2, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x + self.b * x**2 + self.c

class Polynomial3(nn.Module):
    def __init__(self):
        super(Polynomial3, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x + self.b * x**2 + self.c * x**3 + self.d

class Polynomial5(nn.Module):
    def __init__(self):
        super(Polynomial5, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))
        self.e = nn.Parameter(torch.randn(1))
        self.f = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x + self.b * x**2 + self.c * x**3 + self.d * x**4 + self.e * x**5 + self.f

