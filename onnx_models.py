#Onnx export code is from [labelme annotation tool](https://github.com/labelmeai/efficient-sam). Huge thanks to Kentaro Wada.

import torch
import torch.nn.functional as F


class OnnxEfficientSam(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def decoder_max_num_input_points(self):
        return self.model.decoder_max_num_input_points

    @property
    def image_encoder(self):
        return self.model.image_encoder

    @property
    def get_image_embeddings(self):
        return self.model.get_image_embeddings

    @property
    def prompt_encoder(self):
        return self.model.prompt_encoder

    @property
    def mask_decoder(self):
        return self.model.mask_decoder

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
    ):
        batch_size, _, input_h, input_w = batched_images.shape
        image_embeddings = self.get_image_embeddings(batched_images)
        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h,
            output_w=input_w,
        )

    def get_rescaled_pts(
        self, batched_points: torch.Tensor, input_h: int, input_w: int
    ):
        return torch.stack(
            [
                batched_points[..., 0] * self.image_encoder.img_size / input_w,
                batched_points[..., 1] * self.image_encoder.img_size / input_h,
            ],
            dim=-1,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ):
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(
            batched_points, input_h, input_w
        )

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
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks,
                (output_h, output_w),
                # NOTE: "bicubic" is inefficient on onnx
                mode="bilinear",
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
        return output_masks, iou_predictions, low_res_masks


class OnnxEfficientSamEncoder(OnnxEfficientSam):
    def forward(self, batched_images: torch.Tensor):
        return self.model.get_image_embeddings(batched_images)


class OnnxEfficientSamDecoder(OnnxEfficientSam):
    def forward(
        self, image_embeddings, batched_points, batched_point_labels, orig_im_size
    ):
        return self.predict_masks(
            image_embeddings=image_embeddings,
            batched_points=batched_points,
            batched_point_labels=batched_point_labels,
            multimask_output=True,
            input_h=orig_im_size[0],
            input_w=orig_im_size[1],
            output_h=orig_im_size[0],
            output_w=orig_im_size[1],
        )
