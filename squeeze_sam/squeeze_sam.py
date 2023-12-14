# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

from efficient_sam.two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer
from efficient_sam.mlp import MLPBlock

class UnetEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        stride: Tuple[int, int],
        in_chans: int,
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
        self.img_size = img_size
        prev_layer_dim = in_chans
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                patch_embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(stride,stride),
                padding=(0,0),
            ),
            torch.nn.GroupNorm(1, patch_embed_dim),
        )
        prev_layer_dim = patch_embed_dim
        self.down_layers = nn.ModuleList([])
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
        ret.append(x)
        for down_layer in self.down_layers:
            x = down_layer(x)
            ret.append(x)
        return ret


class NullPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        """
        Null encoder that only returns zeros for all inputs. This is useful when the prompts are already encoded earlier in the network.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        coords,
        labels,
    ) -> torch.Tensor:
        batch_size, _, _ = coords.shape
        return torch.zeros(batch_size, 1, self.embed_dim, device=coords.device)


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
        self.norm.qconfig = None
        self.quant_output_normalization = torch.ao.quantization.QuantStub()

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


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        unet_conv_dims: List[int],
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        if num_multimask_outputs > 1:
            self.num_mask_tokens = num_multimask_outputs + 1
        else:
            self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        output_dim_after_upscaling = transformer_dim
        self.unet_upscaling_layers = nn.ModuleList([])
        for unet_layer_dim in unet_conv_dims:
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


        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=output_dim_after_upscaling,
                    num_layers=2,
                    act=activation,
                )
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )



    def forward(
        self,
        image_embeddings: List[torch.Tensor],
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        (
            batch_size,
            max_num_queries,
            sparse_embed_dim_1,
            sparse_embed_dim_2,
        ) = sparse_prompt_embeddings.shape
        sparse_prompt_embeddings = sparse_prompt_embeddings.reshape(
            batch_size * max_num_queries, sparse_embed_dim_1, sparse_embed_dim_2
        )
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        return masks[:, 1:, :], iou_pred[:, 1:]

    def predict_masks(
        self,
        image_embeddings: List[torch.Tensor],
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        final_image_embedding = image_embeddings[-1]
        b, c, h, w = final_image_embedding.shape
        hs, src = self.transformer(final_image_embedding, torch.zeros(1, final_image_embedding.shape[1], h, w, device=final_image_embedding.device), tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)
        # the first Up layer is not meaningful to take in the last embedding from image_embeddings
        # since it takes in the last embedding from the encoder and its derivative from the decoder transformer.
        # Instead it should be a skip connection to the earlier embedding.
        down_conv_embeddings = image_embeddings[:-1]
        down_conv_embeddings.reverse()
        for idx, upscaling_layer in enumerate(self.unet_upscaling_layers):
            upscaled_embedding = upscaling_layer(
                upscaled_embedding, down_conv_embeddings[idx]
            )

        hyper_in_list: List[torch.Tensor] = []
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class SqueezeSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: UnetEncoder,
        prompt_encoder: NullPromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
        user_input_circle_radius: float = 5.0,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (UnetEncoder): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.user_input_circle_radius = user_input_circle_radius
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: List[torch.Tensor],
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """

        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                :, :, : self.decoder_max_num_input_points, :
            ]
            batched_point_labels = batched_point_labels[
                :, :, : self.decoder_max_num_input_points
            ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )
        sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
        iou_predictions = torch.take_along_dim(iou_predictions, sorted_ids, dim=2)
        output_masks = torch.take_along_dim(
            output_masks, sorted_ids[..., None, None], dim=2
        )
        return output_masks, iou_predictions

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )


    @torch.jit.export
    def add_user_inputs_as_separate_channels(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        input_h: int,
        input_w: int,
    ) -> torch.Tensor:
        """
        This function creates a rectangular mask corresponding to the input bounding box and adds as a separate channel.
        If no bounding box is found in the query, mask for the entire image is created by default.
        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, max_num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, max_num_pts]

        Returns:
          A tensor of shape [B, max_num_queries, 4, H, W] after augmenting the bbox_mask as the fourth channel.
        """
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)
        batch_size, _, h, w = batched_images.shape
        max_num_queries = rescaled_batched_points.shape[1]
        device = batched_images.device
        row_ids = torch.arange(h).to(device)
        col_ids = torch.arange(w).to(device)
        row_ids = torch.tile(
            torch.reshape(row_ids, (1, 1, h, 1)),
            (batch_size, max_num_queries, 1, w),
        )
        col_ids = torch.tile(
            torch.reshape(col_ids, (1, 1, 1, w)),
            (batch_size, max_num_queries, h, 1),
        )
        is_bbox = torch.ne(batched_point_labels[:, :, 0], 1).float()
        # Use (0,0) as the top left corner if no bbox is found in the query (is_bbox=False).
        top_left_x = (
            is_bbox * torch.min(rescaled_batched_points[:, :, :2, 0], dim=-1).values
        )
        top_left_y = (
            is_bbox * torch.min(rescaled_batched_points[:, :, :2, 1], dim=-1).values
        )

        # Use (w,h) as the bottom right corner if no bbox is found in the query (is_bbox=False).
        bottom_right_x = (
            is_bbox * torch.max(rescaled_batched_points[:, :, :2, 0], dim=-1).values
            + (1 - is_bbox) * w
        )
        bottom_right_y = (
            is_bbox * torch.max(rescaled_batched_points[:, :, :2, 1], dim=-1).values
            + (1 - is_bbox) * h
        )
        row_within = torch.logical_and(
            torch.ge(row_ids, top_left_y[..., None, None]),
            torch.le(row_ids, bottom_right_y[..., None, None]),
        )
        col_within = torch.logical_and(
            torch.ge(col_ids, top_left_x[..., None, None]),
            torch.le(col_ids, bottom_right_x[..., None, None]),
        )
        bbox_mask = torch.logical_and(row_within, col_within)
        is_point = torch.eq(batched_point_labels, 1).float()[:, :, None, None, :]
        # cols_and_rows is B, Q, h, w, 2
        # rescaled_batched_points is [B, Q, max_num_pts, 2]
        cols_and_rows = torch.stack([col_ids, row_ids], dim=-1)
        user_input_circle_radius = self.user_input_circle_radius
        user_input_pts_mask = torch.le(
            torch.min(
                torch.sum(
                    torch.square(
                        cols_and_rows[:, :, :, :, None, :]
                        - rescaled_batched_points[:, :, None, None, :, :]
                    ),
                    dim=-1,
                )
                * is_point
                + (1 - is_point)
                * (user_input_circle_radius * user_input_circle_radius + 1),
                dim=-1,
            ).values,
            user_input_circle_radius * user_input_circle_radius,
        )
        batched_images = torch.tile(
            batched_images[:, None, :, :, :], [1, max_num_queries, 1, 1, 1]
        )
        return torch.cat(
            [
                batched_images,
                bbox_mask[:, :, None, :, :],
                user_input_pts_mask[:, :, None, :, :],
            ],
            dim=2,
        )


    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> List[torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)


    @torch.jit.export
    def get_image_embeddings_with_early_fusion(
        self, batched_images, batched_points, batched_point_labels, input_h: int, input_w: int
    ) -> List[torch.Tensor]:
        """
        Gets image embeddings after adding bbox and points as separate channels.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        batch_size, _, H, W = batched_images.shape
        max_num_queries = batched_points.shape[1]
        batched_images = self.add_user_inputs_as_separate_channels(
            batched_images, batched_points, batched_point_labels, input_h, input_w
        )
        batched_images = torch.reshape(
            batched_images, [batch_size * max_num_queries, 5, H, W]
        )
        return self.image_encoder(batched_images)



    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape
        image_embeddings = self.get_image_embeddings_with_early_fusion(
            batched_images, batched_points, batched_point_labels, input_h, input_w
        )
        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_squeeze_sam(checkpoint=None):
    img_size = 512
    encoder_patch_embed_dim = 64
    encoder_patch_size = 2
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 256
    decoder_num_heads = 8
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation_fn = nn.ReLU
    normalization_type = "batch_norm"
    normalize_before_activation = False
    decoder_max_num_input_points = 6
    user_input_circle_radius=5.0



    encoder_unet_conv_dims = [64, 128, 128, 128, 128, 164, 256, 256]

    decoder_unet_conv_dims = list(reversed(encoder_unet_conv_dims))[1:]

    image_encoder = UnetEncoder(
        img_size=img_size,
        patch_size=encoder_patch_size,
        stride=encoder_patch_size,
        in_chans=5,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        normalize_before_activation=normalize_before_activation,
        down_conv_dims=encoder_unet_conv_dims,
        act_layer=activation_fn,
    )

    # image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = encoder_unet_conv_dims[-1]

    sam = SqueezeSam(
        image_encoder=image_encoder,
        prompt_encoder=NullPromptEncoder(
            embed_dim=encoder_transformer_output_dim,
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            unet_conv_dims=decoder_unet_conv_dims,
        ),
        user_input_circle_radius=user_input_circle_radius,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"])
    return sam
