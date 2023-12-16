from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from squeeze_sam import EarlyFusion
from squeeze_sam.mask_decoder import MaskDecoder

from squeeze_sam.patch_embed import PatchEmbed
from squeeze_sam.unet_encoder import UnetEncoder
from squeeze_sam.detection_transformer import DetectionTransformer
from squeeze_sam.unet_decoder import UnetDecoder



class SqueezeSam(nn.Module):
    def __init__(
        self,
        encoder_image_size: int,
        encoder_patch_size: int,
        encoder_patch_embed_dim: int,
        activation_fn: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        encoder_unet_conv_dims: List[int],
        decoder_transformer_depth: int,
        decoder_transformer_mlp_dim: int,
        decoder_num_heads: int,
        num_multimask_outputs: int,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        """
        super().__init__()
        # Early fusion of image and user inputs.
        self.early_fusion_model = EarlyFusion(encoder_image_size=encoder_image_size)

        # Patch embedding of the image to lower the resolution.
        self.patch_embed = PatchEmbed(patch_size=encoder_patch_size,
        in_chans=5,
        patch_embed_dim=encoder_patch_embed_dim)
        # UNET encoder.
        self.unet_encoder = UnetEncoder(patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        normalize_before_activation=normalize_before_activation,
        down_conv_dims=encoder_unet_conv_dims,
        act_layer=activation_fn)

        # Transformer that takes the output of the UNET encoder and outputs tokens for estimated IOU and mask.
        # This is a very light weight transformer.
        self.get_iou_predictions_and_mask_tokens = DetectionTransformer(unet_middle_layer_dim=encoder_unet_conv_dims[-1],
        decoder_transformer_depth=decoder_transformer_depth,
        decoder_num_heads=decoder_num_heads,
        decoder_transformer_mlp_dim=decoder_transformer_mlp_dim,
        num_multimask_outputs=num_multimask_outputs,
        activation=activation_fn,
        normalization_type=normalization_type,
        normalize_before_activation=normalize_before_activation)

        # UNET decoder.
        self.unet_decoder = UnetDecoder(
        activation=activation_fn,
        normalization_type=normalization_type,
        normalize_before_activation=normalize_before_activation,
        unet_conv_dims=encoder_unet_conv_dims)

        # The final prediction layer that takes the output of the UNET decoder, tokens (from transformer above) and outputs masks.
        self.mask_decoder=MaskDecoder(unet_middle_layer_dim=encoder_unet_conv_dims[-1],
        unet_final_layer_dim=encoder_unet_conv_dims[0],
        num_multimask_outputs=num_multimask_outputs,
        activation=activation_fn,
        normalization_type=normalization_type,
        normalize_before_activation=normalize_before_activation,
        iou_head_depth=iou_head_depth,
        iou_head_hidden_dim=iou_head_hidden_dim)

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
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
        batch_size, num_masks, num_points, _ = batched_points.shape
        early_fused_image = self.early_fusion_model(batched_images, batched_points, batched_point_labels)
        patch_embeddings = self.patch_embed(early_fused_image)
        encoded_embeddings = self.unet_encoder(patch_embeddings)
        iou_tokens_out, mask_tokens_out, updated_final_encoder_embedding = self.get_iou_predictions_and_mask_tokens(encoded_embeddings[-1])

        encoded_embeddings[-1]=updated_final_encoder_embedding
        final_unet_embedding = self.unet_decoder(encoded_embeddings)
        low_res_mask_logits, iou_pred = self.mask_decoder(mask_tokens_out, iou_tokens_out, final_unet_embedding)
        # The first prediction is ignored (we asked the model to predict 4 masks, but only choose the last 3). This is not ideal.
        low_res_mask_logits = low_res_mask_logits[:, 1:, ...]
        iou_pred = iou_pred[:, 1:]
        output_mask_logits = F.interpolate(
            low_res_mask_logits, (input_h, input_w), mode="bicubic"
        )
        num_output_masks = output_mask_logits.shape[1]
        return output_mask_logits.reshape(batch_size, num_masks, num_output_masks, input_h, input_w), iou_pred.reshape(batch_size, num_masks, num_output_masks)


def build_squeeze_sam_base(checkpoint=None):
    squeeze_sam = SqueezeSam(encoder_image_size=512,
    encoder_patch_size=2,
    encoder_patch_embed_dim=64,
    activation_fn = nn.ReLU,
    normalization_type = "batch_norm",
    normalize_before_activation=False ,
    encoder_unet_conv_dims = [64, 128, 128, 128, 128, 164, 256, 256],
    decoder_transformer_depth = 2,
    decoder_transformer_mlp_dim = 256,
    decoder_num_heads = 8,
    num_multimask_outputs = 3,
    iou_head_depth = 2,
    iou_head_hidden_dim = 256)
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        squeeze_sam.load_state_dict(state_dict["model"])
    return squeeze_sam
