from efficient_sam import build_efficient_sam
from PIL import Image
from torchvision import transforms
import torch
import numpy as np


def get_sam_mask_using_points(img_tensor, pts_sampled, model):
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    max_num_pts = pts_sampled.shape[2]
    pts_labels = torch.ones(1, 1, max_num_pts)
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()


efficient_sam = build_efficient_sam('model_ckpt.pth')
efficient_sam.eval()

# load an image
sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
sample_image_tensor = transforms.ToTensor()(sample_image_np)
# Feed a few (x,y) points in the mask as input.

input_point = np.array([[580, 350], [650, 350]])
input_label = np.array([1, 1])

mask = get_sam_mask_using_points(sample_image_tensor, input_point, efficient_sam)
masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
Image.fromarray(masked_image_np).save("figs/examples/dogs_efficient_sam_mask.png")
