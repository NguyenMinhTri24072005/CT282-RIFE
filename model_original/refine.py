import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model_original.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Custom Activation Modules
# ============================================================

class SoftClampReLU(nn.Module):
    """Soft-Clamp ReLU: f(x) = tau * tanh(ReLU(x) / tau)

    Smooth upper-bounded variant of ReLU. Gradient không bị cắt đột ngột
    ở phần dương, phù hợp cho flow estimation khi cần kiểm soát magnitude.

    Args:
        tau (float): ngưỡng clamp mềm. Mặc định 6.0
                     (output ≈ bão hoà tại ~3*tau)
    """
    def __init__(self, tau: float = 6.0):
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * torch.tanh(torch.relu(x) / self.tau)

    def extra_repr(self) -> str:
        return f'tau={self.tau}'


class SoftClampSiLU(nn.Module):
    """Soft-Clamp SiLU: f(x) = tau * tanh(SiLU(x) / tau)

    Kết hợp tính smooth 2 chiều của SiLU (x * sigmoid(x)) với
    soft-clamping để tránh giá trị bùng nổ ở vùng dương lớn.

    Args:
        tau (float): ngưỡng clamp mềm. Mặc định 6.0
    """
    def __init__(self, tau: float = 6.0):
        super().__init__()
        self.tau = tau
        self._silu = nn.SiLU(inplace=False)  # inplace=False vì cần giá trị gốc cho tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * torch.tanh(self._silu(x) / self.tau)

    def extra_repr(self) -> str:
        return f'tau={self.tau}'



class SmoothPReLU(nn.Module):
    """Smooth-PReLU: f(x) = ((1+a)/2)*x + ((1-a)/2)*sqrt(x^2 + eps^2)
    - a: learnable slope parameter cho từng kênh (khởi tạo 0.25 như PReLU gốc)
    - eps: độ mượt quanh điểm 0, giữ lại thông tin tại x=0 và làm trơn gradient.
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, eps: float = 0.05, a_max: float = 1.0):
        super().__init__()
        self.num_parameters = num_parameters
        self.eps = eps
        self.a_max = a_max #= 1.0  # Giới hạn trên cho a để tránh gradient quá lớn
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(self.weight, -self.a_max, self.a_max) # Giới hạn giá trị a trong khoảng [-a_max, a_max]
        if self.num_parameters > 1:
            w = self.weight.view(1, self.num_parameters, 1, 1)
        term1 = 0.5 * (1.0 + w) * x
        term2 = 0.5 * (1.0 - w) * torch.sqrt(x ** 2 + self.eps ** 2)
        return term1 + term2

    def extra_repr(self) -> str:
        return f'num_parameters={self.num_parameters}, eps={self.eps}'
    
    
    
class OptimizedSmoothPReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25, eps: float = 0.01, a_min: float = 0.0, a_max: float = 0.5):
        super().__init__()
        self.num_parameters = num_parameters
        self.eps = eps
        self.a_min = a_min  # Chặn dưới = 0.0 (không cho âm)
        self.a_max = a_max  # Chặn trên = 0.5
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(self.weight, self.a_min, self.a_max)
        if self.num_parameters > 1:
            w = w.view(1, self.num_parameters, 1, 1)
        term1 = 0.5 * (1.0 + w) * x
        term2 = 0.5 * (1.0 - w) * torch.sqrt(x ** 2 + self.eps ** 2)
        return term1 + term2

# ============================================================
# Activation Function Configuration
# Hỗ trợ: 'prelu' (gốc), 'gelu', 'silu',
#          'soft_clamp_relu', 'soft_clamp_silu'
# Gọi set_activation('gelu') TRƯỚC KHI khởi tạo Model()
# ============================================================
_ACT_FN = 'prelu'

def set_activation(name):
    """Thiết lập hàm kích hoạt cho toàn bộ mạng RIFE.
    Gọi hàm này TRƯỚC KHI khởi tạo Model().
    
    Args:
        name: 'prelu', 'gelu', hoặc 'silu'
    """
    global _ACT_FN
    name = name.lower().strip()
    supported = ['prelu', 'gelu', 'silu', 'soft_clamp_relu', 'soft_clamp_silu', 'smooth_prelu']
    if name not in supported:
        raise ValueError(f"Activation '{name}' không được hỗ trợ. Chọn một trong: {supported}")
    _ACT_FN = name
    print(f"✅ Activation function đã được thiết lập: {_ACT_FN.upper()}")

def get_act_layer(out_planes):
    """Tạo layer activation dựa trên cấu hình hiện tại."""
    if _ACT_FN == 'prelu':
        return nn.PReLU(out_planes)
    elif _ACT_FN == 'smooth_prelu':
        return SmoothPReLU(num_parameters=out_planes, eps=0.05)
    elif _ACT_FN == 'optimized_smooth_prelu':
        return OptimizedSmoothPReLU(num_parameters=out_planes, eps=0.01, a_min=0.0, a_max=0.5)
    elif _ACT_FN == 'gelu':
        return nn.GELU()
    elif _ACT_FN == 'silu':
        return nn.SiLU(inplace=True)
    elif _ACT_FN == 'soft_clamp_relu':
        return SoftClampReLU(tau=6.0)
    elif _ACT_FN == 'soft_clamp_silu':
        return SoftClampSiLU(tau=6.0)
    else:
        return nn.PReLU(out_planes)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        get_act_layer(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        get_act_layer(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(17, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)
