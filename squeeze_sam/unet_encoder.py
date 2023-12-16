import torch.nn as nn
from typing import List, Type
import torch


class Conv2dWithActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act_layer: Type[nn.Module],
        normalization_type: str,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
            ),
            act_layer(),
        )
        assert (
            normalization_type == "layer_norm"
            or normalization_type == "batch_norm"
            or normalization_type == "sync_batch_norm"
            or normalization_type == "none"
        )
        if normalization_type == "layer_norm":
            self.norm = torch.nn.GroupNorm(1, out_channels)
        elif normalization_type == "batch_norm":
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif normalization_type == "sync_batch_norm":
            self.norm = torch.nn.SyncBatchNorm(out_channels)
        elif normalization_type == "none":
            self.norm = torch.nn.Identity()
        else:
            raise ValueError(f"normalization type {normalization_type} not supported")

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x



class DoubleConv(nn.Module):
    """(convolution => [GELU] => LN) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv2dWithActivation(
                in_channels,
                out_channels,
                kernel_size=3,
                act_layer=act_layer,
                normalization_type=normalization_type,
            ),
            Conv2dWithActivation(
                out_channels,
                out_channels,
                kernel_size=3,
                act_layer=act_layer,
                normalization_type=normalization_type,
            ),
        )

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                act_layer,
                normalization_type,
                normalize_before_activation,
            ),
        )

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

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



class UnetEncoder(nn.Module):
    def __init__(
        self,
        patch_embed_dim: int,
        normalization_type: str,
        normalize_before_activation: bool,
        down_conv_dims: List[int],
        act_layer: Type[nn.Module],
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.quant_input = torch.ao.quantization.QuantStub()
        self.down_layers = nn.ModuleList([])
        self.dequant_outputs = nn.ModuleList([])
        self.down_layers.append(
            DoubleConv(
                patch_embed_dim,
                down_conv_dims[0],
                act_layer,
                normalization_type,
                normalize_before_activation,
            )
        )
        self.dequant_outputs.append(torch.ao.quantization.DeQuantStub())
        prev_layer_dim = down_conv_dims[0]
        for idx in range(len(down_conv_dims) - 1):
            self.down_layers.append(
                Down(
                    down_conv_dims[idx],
                    down_conv_dims[idx + 1],
                    act_layer,
                    normalization_type,
                    normalize_before_activation,
                )
            )
            prev_layer_dim = down_conv_dims[idx + 1]
            self.dequant_outputs.append(torch.ao.quantization.DeQuantStub())
        self.image_embedding_size = prev_layer_dim



    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ret = torch.jit.annotate(List[torch.Tensor], [])
        x = self.quant_input(x)
        ret.append(x)
        for down_layer, dequant_embedding in zip(self.down_layers,self.dequant_outputs):
            x = down_layer(x)
            ret.append(dequant_embedding(x))
        return ret
