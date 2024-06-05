import cv2
import zipfile
from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)
from segment_anything import sam_model_registry

# from pre_process import process_folders, convert_to_rgb
import nibabel as nib
import os
from visualizationTools import (
    show_mask,
    show_points,
    show_box,
    show_anns_ours,
    run_ours_box_or_points,
)
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from EvaluationMetrics import compute_iou, compute_dice
import torch.nn.functional as F
from skimage import io, transform
import pandas as pd

device = "cuda:0"


# copied code from https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def convert_to_rgb(hu_image, min_hu=-1200, max_hu=3000):
    hu_image_clipped = np.clip(hu_image, min_hu, max_hu)
    np_img_normalized = (hu_image_clipped - min_hu) / (max_hu - min_hu) * 255
    np_img_normalized = np_img_normalized.astype(np.uint8)

    rgb_image = np.stack([np_img_normalized] * 3, axis=-1)
    return rgb_image


def run_ours_box_or_points(image_np, box, pts_labels, esam, medsam):
    H, W, _ = image_np.shape
    box_np = np.array([box])
    img_1024 = transform.resize(
        image_np, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = esam.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
        medsam_seg = medsam_inference(medsam, image_embedding, box_1024, H, W)
    return medsam_seg


def process_folders(
    training_file_path,
    label_file_path,
    output_folder,
    append_to_csv=True,
    efficient_sam_vitt_model=None,
    efficient_sam_vits_model=None,
    medsam_model=None,
):
    img_sticker = os.path.basename(training_file_path).split(".")[0]

    training_img = nib.load(training_file_path)
    label_img = nib.load(label_file_path)
    training_data = training_img.get_fdata()
    label_data = label_img.get_fdata()
    # Print the shape of the label_img
    print(f"Shape of label_img: {label_data.shape}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    iou_dice_data = []

    for slice_index in range(label_data.shape[2]):
        label_slice = label_data[:, :, slice_index]

        if np.any(label_slice > 0):
            training_slice = training_data[:, :, slice_index].T
            rgb_image = convert_to_rgb(training_slice, min_hu=-1200, max_hu=3000)

            # Calculate bounding box
            x1, y1, w, h = cv2.boundingRect(label_slice.astype(np.uint8))
            x2 = x1 + w
            y2 = y1 + h
            enlarge_by = 6

            x1 = x1 - enlarge_by
            x2 = x2 + enlarge_by
            y1 = y1 - enlarge_by
            y2 = y2 + enlarge_by

            fig, ax = plt.subplots(1, 4, figsize=(50, 30))
            input_point = np.array([[x1, y1], [x2, y2]])
            input_label = np.array([2, 3])

            show_points(input_point, input_label, ax[0])
            show_box([x1, y1, x2, y2], ax[0])
            ax[0].imshow(rgb_image)

            ax[1].imshow(rgb_image)
            mask_efficient_sam_vitt = run_ours_box_or_points(
                rgb_image,
                [x1, y1, x2, y2],
                input_label,
                efficient_sam_vitt_model,
                medsam_model,
            )
            show_anns_ours(mask_efficient_sam_vitt, ax[1])
            iou_vitt = compute_iou(mask_efficient_sam_vitt, label_slice)
            dice_vitt = compute_dice(mask_efficient_sam_vitt, label_slice)
            ax[1].title.set_text(
                f"EfficientSAM (VIT-tiny)\nIoU: {iou_vitt:.4f}, Dice: {dice_vitt:.4f}"
            )
            ax[1].axis("off")

            ax[2].imshow(rgb_image)
            mask_efficient_sam_vits = run_ours_box_or_points(
                rgb_image,
                [x1, y1, x2, y2],
                input_label,
                efficient_sam_vits_model,
                medsam_model,
            )
            show_anns_ours(mask_efficient_sam_vits, ax[2])
            iou_vits = compute_iou(mask_efficient_sam_vits, label_slice)
            dice_vits = compute_dice(mask_efficient_sam_vits, label_slice)
            ax[2].title.set_text(
                f"EfficientSAM (VIT-small)\nIoU: {iou_vits:.4f}, Dice: {dice_vits:.4f}"
            )
            ax[2].axis("off")

            ax[3].imshow(label_slice, cmap="gray")
            ax[3].title.set_text(f"Label Slice {slice_index}")
            ax[3].axis("off")

            slice_filename = os.path.join(
                output_folder, f"{img_sticker}_segmented_slice_{slice_index}.png"
            )
            plt.savefig(slice_filename)
            plt.close(fig)

            iou_dice_data.append(
                {
                    "Image Sticker": img_sticker,
                    "Slice Index": slice_index,
                    "Model": "EfficientSAM (VIT-tiny)",
                    "IoU": iou_vitt,
                    "Dice": dice_vitt,
                }
            )
            iou_dice_data.append(
                {
                    "Image Sticker": img_sticker,
                    "Slice Index": slice_index,
                    "Model": "EfficientSAM (VIT-small)",
                    "IoU": iou_vits,
                    "Dice": dice_vits,
                }
            )

    iou_dice_df = pd.DataFrame(iou_dice_data)
    csv_path = os.path.join(output_folder, "iou_dice_scores.csv")
    if append_to_csv and os.path.exists(csv_path):
        existing_iou_dice_df = pd.read_csv(csv_path)
        iou_dice_df = pd.concat([existing_iou_dice_df, iou_dice_df], ignore_index=True)
    iou_dice_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", "r") as zip_ref:
        zip_ref.extractall("weights")

    efficient_sam_vits_model = build_efficient_sam_vits()
    efficient_sam_vits_model.to(device)
    efficient_sam_vits_model.eval()

    efficient_sam_vitt_model = build_efficient_sam_vitt()
    efficient_sam_vitt_model.to(device)
    efficient_sam_vitt_model.eval()

    MedSAM_CKPT_PATH = "medsam_vit_b.pth"

    medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    # load image and label
    training_data_folder = "dataset/Task06_Lung/imagesTr"
    label_data_folder = "dataset/Task06_Lung/labelsTr"
    output_folder = "dataset/Task06_Lung/output"

    for filename in os.listdir(training_data_folder):
        if filename.endswith(".nii.gz") and not filename.startswith("._"):
            training_file_path = os.path.join(training_data_folder, filename)
            label_file_path = os.path.join(label_data_folder, filename)
            process_folders(
                training_file_path,
                label_file_path,
                output_folder,
                efficient_sam_vits_model=efficient_sam_vits_model,
                efficient_sam_vitt_model=efficient_sam_vitt_model,
                medsam_model=medsam_model,
            )
        else:
            print(
                f"One of the paths does not exist: {training_data_folder}, {label_data_folder}"
            )
