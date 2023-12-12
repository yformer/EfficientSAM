from efficient_sam.efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

models = {}
# Build the VIT-tiny model.
models['vitt'] = build_efficient_sam_vitt()


# # Build the VIT-small model.
# models['vits'] = build_efficient_sam_vits()

# load an image
sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
sample_image_tensor = transforms.ToTensor()(sample_image_np)
# Feed a few (x,y) points in the mask as input.

input_points = torch.tensor([[[[580, 350], [650, 350]]]])
input_labels = torch.tensor([[[1, 1]]])

# Run inference for both vitt and vits based models.
for model_name, efficient_sam in models.items():
    print('Running inference using ', model_name)
    efficient_sam.half()
    predicted_logits, predicted_iou = efficient_sam(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
    Image.fromarray(masked_image_np).save(f"figs/examples/dogs_efficient_sam_{model_name}_mask.png")
