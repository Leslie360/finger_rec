import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from config import PHYSICAL_MIN, PHYSICAL_MAX, DEVICE_NOISE_STD, NUM_CLASSES

class OrganicSynapseConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        
        self.phys_min = PHYSICAL_MIN
        self.phys_max = PHYSICAL_MAX
        self.noise_std = DEVICE_NOISE_STD
        
    def forward(self, input):
        # 1. Map to Physical
        w_norm = (self.weight + 1) / 2
        w_phys = w_norm * (self.phys_max - self.phys_min) + self.phys_min
        
        # 2. Inject Noise (Training only)
        if self.training:
            noise = torch.randn_like(w_phys) * self.noise_std
            w_phys = w_phys + noise
            
        # 3. Map back
        w_norm_noisy = (w_phys - self.phys_min) / (self.phys_max - self.phys_min)
        w_math_noisy = w_norm_noisy * 2 - 1
        
        return self._conv_forward(input, w_math_noisy, self.bias)

def get_organic_resnet18(pretrained=False):
    model = resnet18(weights='DEFAULT' if pretrained else None) # 修复：使用 weights 参数避免警告
    
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                new_conv = OrganicSynapseConv(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                new_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_conv.bias.data = child.bias.data.clone()
                setattr(module, name, new_conv)
            else:
                replace_layers(child)
                
    replace_layers(model)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES) 
    
    return model