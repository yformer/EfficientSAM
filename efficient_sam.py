# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from mlp import MLPBlock
from efficient_sam_decoder import MaskDecoder, NullPromptEncoder, PromptEncoder
from efficient_sam_encoder import ImageEncoderViT


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        p_dropout: float,
        activation: Type[nn.Module],
        normalize_before_activation: bool,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            curr_layer = TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                p_dropout=p_dropout,
                activation=activation,
                normalize_before_activation=normalize_before_activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            self.layers.append(curr_layer)

        self.final_attn_token_to_image = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            p_dropout,
            downsample_rate=attention_downsample_rate,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        self.norm_final_attn.qconfig = None

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for idx, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        p_dropout: float,
        activation: Type[nn.Module],
        normalize_before_activation: bool,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = AttentionForTwoWayAttentionBlock(
            embedding_dim, num_heads, p_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm1.qconfig = None
        self.dropout1 = nn.Dropout(p_dropout)

        self.cross_attn_token_to_image = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            p_dropout,
            downsample_rate=attention_downsample_rate,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm2.qconfig = None
        self.dropout2 = nn.Dropout(p_dropout)

        self.quant_input_mlp = torch.ao.quantization.QuantStub()
        self.mlp = MLPBlock(
            embedding_dim,
            mlp_dim,
            embedding_dim,
            1,
            activation,
        )
        self.dequant_output_mlp = torch.ao.quantization.DeQuantStub()

        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm3.qconfig = None
        self.dropout3 = nn.Dropout(p_dropout)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.norm4.qconfig = None
        self.cross_attn_image_to_token = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            p_dropout,
            downsample_rate=attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    # Refer https://www.internalfb.com/code/fbsource/[1caffe8318fe6916923c9eaa984221deb72752f4]/fbcode/deeplearning/projects/segmenting_everything/segment_everything/modeling/transformer/transformer.py
    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if not self.skip_first_layer_pe:
            queries = queries + query_pe
        attn_out = self.self_attn(q=queries, k=queries, v=queries)
        queries = queries + self.dropout1(attn_out)
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + self.dropout2(attn_out)
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.dequant_output_mlp(self.mlp(self.quant_input_mlp(queries)))
        queries = queries + self.dropout3(mlp_out)
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class AttentionForTwoWayAttentionBlock(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.q_dropout = nn.Dropout(proj_dropout)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_dropout = nn.Dropout(proj_dropout)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_dropout = nn.Dropout(proj_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.out_proj.qconfig = None
        self.out_dropout = nn.Dropout(proj_dropout)
        self._reset_parameters()

        # Quant/Dequant stubs

        self.quant_q = torch.ao.quantization.QuantStub()
        self.quant_k = torch.ao.quantization.QuantStub()
        self.quant_v = torch.ao.quantization.QuantStub()

        self.dequant_q = torch.ao.quantization.DeQuantStub()
        self.dequant_k = torch.ao.quantization.DeQuantStub()
        self.dequant_v = torch.ao.quantization.DeQuantStub()

    def _reset_parameters(self) -> None:
        # The fan_out is incorrect, but matches pytorch's initialization
        # for which qkv is a single 3*embedding_dim x embedding_dim matrix
        fan_in = self.embedding_dim
        fan_out = 3 * self.internal_dim
        # Xavier uniform with our custom fan_out
        bnd = math.sqrt(6 / (fan_in + fan_out))
        nn.init.uniform_(self.q_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.k_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.v_proj.weight, -bnd, bnd)
        # out_proj.weight is left with default initialization, like pytorch attention
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_dropout(self.q_proj(self.quant_q(q)))
        k = self.k_dropout(self.k_proj(self.quant_k(k)))
        v = self.v_dropout(self.v_proj(self.quant_v(v)))

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        q = self.dequant_q(q)
        k = self.dequant_k(k)
        v = self.dequant_v(v)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_dropout(self.out_proj(out))
        return out


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        fusion_type: str,
        apply_softmax_on_iou_predictions: bool,
        user_input_circle_radius: float,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          fusion_type: Either 'early', 'late' or 'hybrid'.
          apply_softmax_on_iou_predictions: Whether to predict classification probabilities instead of IOU scores.
          user_input_circle_radius: Radius of the circle used to encode user points in early fusion.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.fusion_type = fusion_type
        self.apply_softmax_on_iou_predictions = apply_softmax_on_iou_predictions
        self.user_input_circle_radius = user_input_circle_radius
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )
        self.H = -1
        self.W = -1

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: List[torch.Tensor],
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
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
        rescaled_batched_points = self.get_rescaled_pts(batched_points)

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
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        if self.apply_softmax_on_iou_predictions:
            iou_predictions = F.softmax(iou_predictions, dim=-1)
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

        return output_masks, torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )

    def get_rescaled_pts(self, batched_points):
        assert (
            self.H > 0 and self.W > 0
        ), "preprocess must be called before calling this function."
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / self.W,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / self.H,
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
        assert (
            self.H > 0 and self.W > 0
        ), "preprocess must be called before calling this function."
        rescaled_batched_points = self.get_rescaled_pts(batched_points)
        batch_size, _, h, w = batched_images.shape
        max_num_queries = rescaled_batched_points.shape[1]
        device = batched_images.device
        row_ids = torch.arange(h, device=device)
        col_ids = torch.arange(w, device=device)
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
        self, batched_images, batched_points, batched_point_labels
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
            batched_images, batched_points, batched_point_labels
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
        batch_size, _, _, _ = batched_images.shape
        if self.fusion_type == "early" or self.fusion_type == "hybrid":
            image_embeddings = self.get_image_embeddings_with_early_fusion(
                batched_images, batched_points, batched_point_labels
            )
        else:
            image_embeddings = self.get_image_embeddings(batched_images)

        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            output_h=self.H if scale_to_original_image_size else -1,
            output_w=self.W if scale_to_original_image_size else -1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        self.H = x.shape[2]
        self.W = x.shape[3]
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

    def get_fused_model(self):
        module_names = []
        types = []
        layers_fused = []
        for name, m in self.named_modules():
            module_names.append(name)
            layers_fused.append(False)
            if isinstance(m, nn.Conv2d):
                types.append("conv")
            elif isinstance(m, nn.Linear):
                types.append("linear")
            elif isinstance(m, nn.BatchNorm2d):
                types.append("bn")
            elif isinstance(m, nn.ReLU):
                types.append("relu")
            else:
                types.append("unknown")
        # print(module_names)
        # print(types)
        num_modules = len(module_names)
        modules_to_fuse = []
        for idx in range(num_modules - 1):
            if (
                idx <= num_modules - 3
                and types[idx] == "conv"
                and types[idx + 1] == "bn"
                and types[idx + 2] == "relu"
            ):
                modules_to_fuse.append(
                    [
                        str(module_names[idx]),
                        str(module_names[idx + 1]),
                        str(module_names[idx + 2]),
                    ]
                )
                layers_fused[idx + 1] = True
            elif types[idx] == "conv" and types[idx + 1] == "bn":
                modules_to_fuse.append(
                    [str(module_names[idx]), str(module_names[idx + 1])]
                )
                layers_fused[idx + 1] = True
            elif types[idx] == "conv" and types[idx + 1] == "relu":
                modules_to_fuse.append(
                    [str(module_names[idx]), str(module_names[idx + 1])]
                )
                layers_fused[idx + 1] = True
            elif types[idx] == "linear" and types[idx + 1] == "relu":
                modules_to_fuse.append(
                    [str(module_names[idx]), str(module_names[idx + 1])]
                )
                layers_fused[idx + 1] = True
        # print(modules_to_fuse)

        if len(modules_to_fuse):
            return torch.ao.quantization.fuse_modules(self, modules_to_fuse)
        else:
            return self


def build_efficient_sam(checkpoint=None, device='cpu'):
    img_size = 1024
    encoder_patch_size = 16
    encoder_patch_embed_dim = 192
    encoder_patch_embed_apply_norm = False
    encoder_unet_conv_dims = []
    encoder_depth = 12
    encoder_num_heads = 3
    encoder_use_rel_pos = False
    encoder_mlp_ratio = 4.0
    encoder_drop_path_rate = 0.4
    encoder_neck_dims = [256, 256]
    encoder_window_size = 0
    encoder_global_attn_indexes = []
    encoder_apply_feature_pyramid = False
    encoder_use_metanet = False
    encoder_metanet_name = "metanetv6_vit_237M_wopool"
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_p_dropout = 0.1
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    fusion_type = "late"
    apply_softmax_on_iou_predictions = False
    user_input_circle_radius = 5.0
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False
    share_hypernetwork_mlp_weights = False

    assert (
        img_size
        % (
            (encoder_patch_size if encoder_patch_size > 0 else 1)
            * (int(2 ** len(encoder_unet_conv_dims)) if encoder_unet_conv_dims else 1)
        )
        == 0
    ), "image size should be divisible by patch_size * (2**len(patch_down_conv_dims))"

    decoder_unet_conv_dims = list(reversed(encoder_unet_conv_dims))[1:]
    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    assert fusion_type == "early" or fusion_type == "late" or fusion_type == "hybrid"
    early_fusion = fusion_type == "early" or fusion_type == "hybrid"

    late_fusion = fusion_type == "late" or fusion_type == "hybrid"

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=5 if early_fusion else 3,
        patch_embed_dim=encoder_patch_embed_dim,
        patch_embed_apply_norm=encoder_patch_embed_apply_norm,
        normalization_type=normalization_type,
        unet_conv_dims=encoder_unet_conv_dims,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        use_rel_pos=encoder_use_rel_pos,
        mlp_ratio=encoder_mlp_ratio,
        drop_path_rate=encoder_drop_path_rate,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
        normalize_before_activation=normalize_before_activation,
        window_size=encoder_window_size,
        global_attn_indexes=encoder_global_attn_indexes,
        apply_feature_pyramid=encoder_apply_feature_pyramid,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        )
        if late_fusion
        else NullPromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                p_dropout=decoder_p_dropout,
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
            upscaling_layer_dims=decoder_upscaling_layer_dims,
            share_hypernetwork_mlp_weights=share_hypernetwork_mlp_weights,
        ),
        fusion_type=fusion_type,
        apply_softmax_on_iou_predictions=apply_softmax_on_iou_predictions,
        user_input_circle_radius=user_input_circle_radius,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f,map_location='cpu')
        sam.load_state_dict(state_dict['model'])
    sam.to(torch.device(device))
    return sam



sam = build_efficient_sam('model_ckpt.pth')

print(sam)
