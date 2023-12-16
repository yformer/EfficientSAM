import torch.nn as nn
from typing import Type, Tuple
import torch
from efficient_sam.mlp import MLPBlock

class MaskDecoder(nn.Module):
    def __init__(
        self,
        unet_middle_layer_dim: int,
        unet_final_layer_dim: int,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
    ) -> None:
        """
        Given token embeddings from the transformer and the final UNET embedding, this predicts masks and IOU scores.
        """
        super().__init__()
        self.unet_middle_layer_dim = unet_middle_layer_dim
        self.num_mask_tokens = num_multimask_outputs + 1
        self.dequant_upscaled_embedding = torch.ao.quantization.DeQuantStub()
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=unet_middle_layer_dim,
                    hidden_dim=unet_middle_layer_dim,
                    output_dim=unet_final_layer_dim,
                    num_layers=2,
                    act=activation,
                )
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLPBlock(
            input_dim=unet_middle_layer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )

    @torch.jit.export
    def forward(
        self,
        mask_tokens_out: torch.Tensor,
        iou_tokens_out: torch.Tensor,
        final_unet_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        hyper_in_list: List[torch.Tensor] = []
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = final_unet_embedding.shape
        masks = (hyper_in @ final_unet_embedding.view(b, c, h * w)).view(b, -1, h, w)
        estimated_iou_scores = self.iou_prediction_head(iou_tokens_out)
        return masks, estimated_iou_scores
