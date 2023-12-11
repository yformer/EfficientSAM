# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLPBlock
from efficient_sam_decoder import ConvTranspose2dUsingLinear, DoubleConv, Down

def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        patch_embed_apply_norm: bool,
        normalization_type: str,
        unet_conv_dims: List[int],
        depth: int,
        num_heads: int,
        use_rel_pos: bool,
        mlp_ratio: float,
        drop_path_rate: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
        normalize_before_activation: bool,
        window_size: int,
        global_attn_indexes: Tuple[int, ...] = (),
        apply_feature_pyramid: bool = False,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.img_size = img_size
        assert (
            img_size
            % (
                (patch_size if patch_size > 0 else 1)
                * (int(2 ** (len(unet_conv_dims) - 1)) if unet_conv_dims else 1)
            )
            == 0
        ), "image size should be divisible by patch_size * (2**len(unet_conv_dims))"
        self.image_embedding_size = (
            img_size
            // (
                (patch_size if patch_size > 0 else 1)
                * (int(2 ** (len(unet_conv_dims) - 1)) if unet_conv_dims else 1)
            )
            * (4 if apply_feature_pyramid else 1)
        )
        self.transformer_output_dim = ([patch_embed_dim] + unet_conv_dims + neck_dims)[
            -1
        ]

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            in_chans=in_chans,
            embed_dim=patch_embed_dim,
            patch_embed_apply_norm=patch_embed_apply_norm,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            down_conv_dims=unet_conv_dims,
            act_layer=act_layer,
        )

        if len(unet_conv_dims):
            patch_embed_dim = unet_conv_dims[-1]

        transformer_image_dims = img_size // (
            (patch_size if patch_size > 0 else 1)
            * int(2 ** (max(0, len(unet_conv_dims) - 1)))
        )

        if not len(unet_conv_dims):
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, transformer_image_dims, transformer_image_dims, patch_embed_dim
                )
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=patch_embed_dim,
                num_heads=num_heads,
                use_rel_pos=use_rel_pos,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                normalize_before_activation=normalize_before_activation,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        in_channels = patch_embed_dim
        if apply_feature_pyramid:
            # Applied here: https://www.internalfb.com/code/fbsource/fbcode/vision/fair/detectron2/detectron2/modeling/backbone/vit.py?lines=363-503
            # It is applied to some internal model, but not all. The external code does not have this as the VIT-B/H models do not apply feature pyramid.
            assert len(unet_conv_dims) == 0
            self.feature_pyramid = nn.Sequential(
                ConvTranspose2dUsingLinear(
                    patch_embed_dim,
                    patch_embed_dim // 2,
                    act_layer,
                ),
                ConvTranspose2dUsingLinear(
                    patch_embed_dim,
                    patch_embed_dim // 2,
                    torch.nn.Identity,
                ),
            )
            in_channels = patch_embed_dim // 4
        else:
            self.feature_pyramid = nn.Identity()

        self.neck_layers = nn.ModuleList([])
        for out_channels in neck_dims:
            if normalization_type == "layer_norm":
                first_norm = torch.nn.GroupNorm(1, out_channels)
                second_norm = torch.nn.GroupNorm(1, out_channels)
            elif normalization_type == "batch_norm":
                first_norm = torch.nn.BatchNorm2d(out_channels)
                second_norm = torch.nn.BatchNorm2d(out_channels)
            elif normalization_type == "sync_batch_norm":
                first_norm = torch.nn.SyncBatchNorm(out_channels)
                second_norm = torch.nn.SyncBatchNorm(out_channels)
            else:
                raise ValueError(
                    f"normalization type {normalization_type} not supported"
                )
            self.neck_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    first_norm,
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    second_norm,
                )
            )
            in_channels = out_channels
        self.apply(self._init_weights)

        # quant / dequant stubs for quantization
        # self.quant_input = torch.ao.quantization.QuantStub()
        self.dequant_all = torch.ao.quantization.DeQuantStub()
        self.dequant_final = torch.ao.quantization.DeQuantStub()
        self.quant_embed = torch.ao.quantization.QuantStub()
        self.dequant_embed = torch.ao.quantization.DeQuantStub()

    def get_image_embedding_size(self) -> int:
        return self.image_embedding_size

    def get_transformer_output_dim(self) -> int:
        return self.transformer_output_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            c2_xavier_fill(m)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        # x = self.quant_input(x)
        # This gets all the patch embeddings by iteratively downsampling the image.
        all_patch_embeddings = self.patch_embed(
            x
        )  # The output of this will be quantized
        final_embeddings = all_patch_embeddings[-1].permute(0, 2, 3, 1)
        if self.pos_embed is not None:
            final_embeddings = self.dequant_embed(final_embeddings)
            final_embeddings = final_embeddings + self.pos_embed
            final_embeddings = self.quant_embed(final_embeddings)

        for blk in self.blocks:
            final_embeddings = blk(final_embeddings)
        final_embeddings = final_embeddings.permute(0, 3, 1, 2)
        final_embeddings = self.feature_pyramid(final_embeddings)
        for neck_layer in self.neck_layers:
            final_embeddings = neck_layer(final_embeddings)
        final_embeddings = self.dequant_final(final_embeddings)
        all_patch_embeddings = [
            self.dequant_all(embeddings) for embeddings in all_patch_embeddings
        ]

        return all_patch_embeddings[:-1] + [final_embeddings]


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_rel_pos: bool,
        mlp_ratio: float,
        drop_path_rate: float,
        act_layer: Type[nn.Module],
        normalize_before_activation: bool,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.window_size = window_size
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.drop_path = (
            nn.Dropout(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            output_dim=dim,
            num_layers=1,
            act=act_layer,
        )

        # quant / dequant functionals
        self.shortcut_add = torch.ao.nn.quantized.FloatFunctional()
        self.act_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x)

        x = self.shortcut_add.add(shortcut, self.drop_path(x))
        x = self.act_add.add(x, self.drop_path(self.mlp(self.norm2(x))))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_rel_pos: bool,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        assert (
            input_size is not None
        ), "Input size must be provided if using relative positional encoding."
        # initialize relative positional embeddings
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
        else:
            self.rel_pos_h = torch.zeros(2 * input_size[0] - 1, head_dim)
            self.rel_pos_w = torch.zeros(2 * input_size[0] - 1, head_dim)

        # Quant/Dequant stubs
        self.quant_attn_output = torch.ao.quantization.QuantStub()
        self.dequant_q = torch.ao.quantization.DeQuantStub()
        self.dequant_k = torch.ao.quantization.DeQuantStub()
        self.dequant_v = torch.ao.quantization.DeQuantStub()
        # Functionals
        self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # attn = (q * self.scale) @ k.transpose(-2, -1)
        q = self.q_scaling_product.mul_scalar(q, self.scale)

        # Leaving the quantized zone here
        q = self.dequant_q(q)
        k = self.dequant_k(k)
        v = self.dequant_v(v)
        attn = q @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.quant_attn_output(x)
        x = self.proj(x)

        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        in_chans: int,
        embed_dim: int,
        patch_embed_apply_norm: bool,
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

        prev_layer_dim = in_chans
        if embed_dim > 0 and kernel_size[0] > 0:

            if patch_embed_apply_norm:
                curr_norm = torch.nn.GroupNorm(1, embed_dim)
            else:
                curr_norm = nn.Identity()

            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    embed_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                curr_norm,
            )
            prev_layer_dim = embed_dim
        else:
            self.proj = nn.Identity()

        self.proj.qconfig = None

        self.quant_input = torch.ao.quantization.QuantStub()

        self.down_layers = nn.ModuleList([])
        if len(down_conv_dims):
            self.down_layers.append(
                DoubleConv(
                    prev_layer_dim,
                    down_conv_dims[0],
                    act_layer,
                    normalization_type,
                    normalize_before_activation,
                )
            )
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ret = torch.jit.annotate(List[torch.Tensor], [])
        x = self.proj(x)
        x = self.quant_input(x)
        ret.append(x)
        for down_layer in self.down_layers:
            x = down_layer(x)
            ret.append(x)
        return ret


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
