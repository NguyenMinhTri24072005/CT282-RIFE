import torch
import torch.nn as nn

class SoftClampReLU(nn.Module):
    def __init__(self, tau: float = 6.0):
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * torch.tanh(torch.relu(x) / self.tau)

class SoftClampSiLU(nn.Module):
    def __init__(self, tau: float = 6.0):
        super().__init__()
        self.tau = tau
        self._silu = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * torch.tanh(self._silu(x) / self.tau)

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

class OptimizedSmoothPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25, eps=0.01, a_min=0.0, a_max=0.5):
        super().__init__()
        self.num_parameters = num_parameters
        self.eps = eps
        self.a_min = a_min
        self.a_max = a_max
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(self.weight, self.a_min, self.a_max)
        if self.num_parameters > 1:
            w = w.view(1, self.num_parameters, 1, 1)
        term1 = 0.5 * (1.0 + w) * x
        term2 = 0.5 * (1.0 - w) * torch.sqrt(x ** 2 + self.eps ** 2)
        return term1 + term2

def get_activation(act_name, num_parameters=1):
    act = act_name.lower().strip()
    if act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'smooth_prelu':
        return SmoothPReLU(num_parameters=num_parameters, eps=0.05)
    elif act == 'optimized_smooth_prelu':
        return OptimizedSmoothPReLU(num_parameters=num_parameters, eps=0.01, a_min=0.0, a_max=0.5)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'silu':
        return nn.SiLU()
    elif act == 'soft_clamp_relu':
        return SoftClampReLU(tau=6.0)
    elif act == 'soft_clamp_silu':
        return SoftClampSiLU(tau=6.0)
    else:
        return nn.PReLU(num_parameters)
