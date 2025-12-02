import torch
import torch.nn as nn
import torch.nn.functional as F

class RainGate(nn.Module):
    """
        A tiny CNN that predicts a per-pixel wet probability map from LR+geo conditions
        Expects input [B, C_in, H, W]; outputs logits [B, 1, H, W] for wet probability
    """
    def __init__(self, c_in: int, c_hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=c_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=c_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_hidden, 1, kernel_size=1, padding=0, bias=True), # Output logits for wet probability
        )
        # Initialize reasonably
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_uniform_(m.weight, a=0.0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)