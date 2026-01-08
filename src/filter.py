import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMFilter(nn.Module):
    """
    Base class for SRM filters.
    """
    def __init__(self, clamp: bool = True):
        super().__init__()
        self.clamp = clamp

        k1 = torch.tensor(
            [[0, 0, 0, 0, 0],
             [0,-1, 2,-1, 0],
             [0, 2,-4, 2, 0],
             [0,-1, 2,-1, 0],
             [0, 0, 0, 0, 0]], dtype=torch.float32
        ) / 4.0

        k2 = torch.tensor(
            [[-1, 2,-2, 2,-1],
             [ 2,-6, 8,-6, 2],
             [-2, 8,-12,8,-2],
             [ 2,-6, 8,-6, 2],
             [-1, 2,-2, 2,-1]], dtype=torch.float32
        ) / 12.0

        k3 = torch.tensor(
            [[0, 0, 0, 0, 0],
             [0, 0,-1, 0, 0],
             [0,-1, 4,-1, 0],
             [0, 0,-1, 0, 0],
             [0, 0, 0, 0, 0]], dtype=torch.float32
        )

        self.base = torch.stack([k1, k2, k3], dim=0)   # (3,5,5)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class.")

class SRMConv6(SRMFilter):

    def __init__(self, clamp: bool = True):
        super().__init__(clamp=clamp)

        # Build grouped conv weights: out=3, in=1, groups=3 (3 kernels per channel)
        w = self.base[:, None, :, :]
        self.register_buffer("weight", w)
        self.srm_norm = nn.GroupNorm(
            num_groups=3,   # 1ch → 3ch per group
            num_channels=3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale
        x_gray = 0.2989 * x[:,0:1,:,:] + 0.5870 * x[:,1:2,:,:] + 0.1140 * x[:,2:3,:,:]
        y = F.conv2d(x_gray, self.weight, padding=2) 
        if self.clamp:
            y = torch.clamp(y, -3.0, 3.0)

        y = self.srm_norm(y)
        y = torch.cat([x, y], dim=1)  # concatenate input and SRM output
        return y

class SRMConv12(SRMFilter):
    """
    Apply 3 SRM kernels to each RGB channel -> 9ch output.
    - Input:  (B,3,H,W)
    - Output: (B,9,H,W)
    """
    def __init__(self, clamp: bool = True):
        super().__init__(clamp=clamp)

        # Build grouped conv weights: out=9, in=3, groups=3 (3 kernels per channel)
        w = torch.zeros((9, 3, 5, 5), dtype=torch.float32)
        for c in range(3):
            w[c*3:(c+1)*3, c:c+1, :, :] = self.base[:, None, :, :]
        self.register_buffer("weight", w)
        self.logit = nn.Parameter(torch.tensor(-3.0))
        self.srm_norm = nn.GroupNorm(
            num_groups=3,   # 9ch → 3ch per group
            num_channels=9
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # groups=1 because weight already maps 3->9 with channel-specific placement
        y = F.conv2d(x, self.weight, padding=2)
        if self.clamp:
            y = torch.clamp(y, -3.0, 3.0)
        
        y = self.srm_norm(y)
        y = torch.cat([x, y * torch.sigmoid(self.logit)], dim=1)  # concatenate input and SRM output
        return y
