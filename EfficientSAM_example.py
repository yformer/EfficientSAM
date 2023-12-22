from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# from squeeze_sam.build_squeeze_sam import build_squeeze_sam

from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile



models = {}

# Build the EfficientSAM-Ti model.
models['efficientsam_ti'] = build_efficient_sam_vitt()

# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
# Build the EfficientSAM-S model.
models['efficientsam_s'] = build_efficient_sam_vits()

# Build the SqueezeSAM model.
# models['squeeze_sam'] = build_squeeze_sam()

# load an image
sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
sample_image_tensor = transforms.ToTensor()(sample_image_np)
# Feed a few (x,y) points in the mask as input.

input_points = torch.tensor([[[[580, 350], [650, 350]]]])
input_labels = torch.tensor([[[1, 1]]])

# Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
for model_name, model in models.items():
    print('Running inference using ', model_name)
    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
    Image.fromarray(masked_image_np).save(f"figs/examples/dogs_{model_name}_mask.png")
