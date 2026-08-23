import torch
import torch.nn as nn

class SmoothPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25, eps=0.05, a_max=1.0):
        super().__init__()
        self.num_parameters = num_parameters
        self.eps = eps
        self.a_max = a_max
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(self.weight, -self.a_max, self.a_max)
        if self.num_parameters > 1:
            w = w.view(1, self.num_parameters, 1, 1)
        term1 = 0.5 * (1.0 + w) * x
        term2 = 0.5 * (1.0 - w) * torch.sqrt(x ** 2 + self.eps ** 2)
        return term1 + term2

def get_activation(act_name, num_parameters=1):
    if act_name == 'prelu':
        return nn.PReLU(num_parameters)
    elif act_name == 'smooth_prelu':
        return SmoothPReLU(num_parameters=num_parameters, eps=0.05)
    elif act_name == 'gelu':
        return nn.GELU()
    elif act_name == 'silu':
        return nn.SiLU()
    else:
        return nn.PReLU(num_parameters) # Default
