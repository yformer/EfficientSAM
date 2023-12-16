
import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                patch_embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                padding=(0, 0),
            ),
            torch.nn.GroupNorm(1, patch_embed_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
