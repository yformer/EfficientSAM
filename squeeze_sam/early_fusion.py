from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusion(nn.Module):
    def __init__(
        self,
        encoder_image_size,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
        user_input_circle_radius: float = 5.0,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          encoder_image_size (UnetEncoder): Image size to the encoder.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
          user_input_circle_radius: Radius of the circle to encode the points as.
        """
        super().__init__()
        self.encoder_image_size = encoder_image_size
        self.user_input_circle_radius = user_input_circle_radius
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

    def get_rescaled_pts(
        self, batched_points: torch.Tensor, input_h: int, input_w: int
    ):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.encoder_image_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.encoder_image_size / input_h,
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
        rescaled_batched_points = self.get_rescaled_pts(
            batched_points, input_h, input_w
        )
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
        max_num_queries = batched_points.shape[1]
        batched_images = self.preprocess(batched_images)
        early_fused_batched_images =  self.add_user_inputs_as_separate_channels(
            batched_images, batched_points, batched_point_labels, input_h, input_w
        )
        return torch.reshape(
            early_fused_batched_images, [batch_size * max_num_queries, 5, self.encoder_image_size, self.encoder_image_size]
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = F.interpolate(
            x,
            (self.encoder_image_size, self.encoder_image_size),
            mode="bilinear",
        )
        return (x - self.pixel_mean) / self.pixel_std
