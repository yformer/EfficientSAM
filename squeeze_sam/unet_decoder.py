import torch.nn as nn
import torch
from typing import List, Type
from .unet_encoder import DoubleConv
import torch.nn.functional as F

class ConvTranspose2dUsingLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization_type: str):
        super().__init__()
        self.out_channels = out_channels

        self.upsampler_linear = nn.Sequential(
            torch.nn.Linear(in_channels, 4 * out_channels, bias=True), torch.nn.ReLU()
        )
        if normalization_type == "layer_norm":
            self.norm = torch.nn.GroupNorm(1, 4 * out_channels)
        elif normalization_type == "batch_norm":
            self.norm = torch.nn.BatchNorm2d(4 * out_channels)
        elif normalization_type == "sync_batch_norm":
            self.norm = torch.nn.SyncBatchNorm(4 * out_channels)
        else:
            raise ValueError(f"normalization type {normalization_type} not supported")

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, h, w = x.shape
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, [batch_size * h * w, in_channels])
        x = self.upsampler_linear(x)
        x = torch.reshape(x, [batch_size, h, w, 4 * self.out_channels])
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.norm(x)
        x = torch.reshape(x, [batch_size, self.out_channels, 2, 2, h, w])
        x = torch.permute(x, [0, 1, 4, 2, 5, 3])
        return torch.reshape(x, [batch_size, self.out_channels, h * 2, w * 2])



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        x1_in_channels: int,
        x2_in_channels: int,
        out_channels: int,
        act_layer: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
    ):
        super().__init__()
        self.up = ConvTranspose2dUsingLinear(
            x1_in_channels, x1_in_channels // 2, normalization_type
        )
        self.conv = DoubleConv(
            x1_in_channels // 2 + x2_in_channels,
            out_channels,
            act_layer,
            normalization_type,
            normalize_before_activation,
        )

    @torch.jit.export
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        unet_conv_dims: List[int],
    ) -> None:
        """
        A set of upsampling layers as in classical UNET.
        """
        super().__init__()
        self.quant_input = torch.ao.quantization.QuantStub()
        self.unet_upscaling_layers = nn.ModuleList([])
        upsampling_conv_dims = list(reversed(unet_conv_dims))[1:]
        output_dim_after_upscaling = unet_conv_dims[-1]
        for unet_layer_dim in upsampling_conv_dims:
            self.unet_upscaling_layers.append(
                Up(
                    output_dim_after_upscaling,
                    unet_layer_dim,
                    unet_layer_dim,
                    activation,
                    normalization_type,
                    normalize_before_activation,
                )
            )
            output_dim_after_upscaling = unet_layer_dim

        self.quant_input_layers = nn.ModuleList([])
        for idx in range(len(unet_conv_dims)+1):
            self.quant_input_layers.append(torch.ao.quantization.QuantStub())

        self.dequant_output = torch.ao.quantization.DeQuantStub()

    @torch.jit.export
    def forward(
        self,
        image_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""

        # Upscale mask embeddings and predict masks using the mask tokens
        for idx in range(len(image_embeddings)):
            image_embeddings[idx] = self.quant_input_layers[idx](
                image_embeddings[idx]
            )  # quantize
        upscaled_embedding = image_embeddings[-1]
        down_conv_embeddings = image_embeddings[:-1]
        num_down_conv_embeddings = len(down_conv_embeddings)
        for idx, upscaling_layer in enumerate(self.unet_upscaling_layers):
            upscaled_embedding = upscaling_layer(
                upscaled_embedding,
                down_conv_embeddings[num_down_conv_embeddings - 1 - idx],
            )

        return self.dequant_output(upscaled_embedding)
