import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import zipfile
import json
import pandas as pd


from PIL import Image
from torchvision.transforms import ToTensor
from visualizationTools import show_mask, show_points, show_box, show_anns_ours, run_ours_box_or_points
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from EvaluationMetrics import compute_iou, compute_dice
from utils import parse_dataset
efficient_sam_vitt_model = build_efficient_sam_vitt()
efficient_sam_vitt_model.eval()

# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
efficient_sam_vits_model = build_efficient_sam_vits()
efficient_sam_vits_model.eval()


def convert_to_rgb(hu_image, min_hu=-1200, max_hu=3000):
    hu_image_clipped = np.clip(hu_image, min_hu, max_hu)
    np_img_normalized = (hu_image_clipped - min_hu) / (max_hu - min_hu) * 255
    np_img_normalized = np_img_normalized.astype(np.uint8)

    rgb_image = np.stack([np_img_normalized] * 3, axis=-1)
    return rgb_image


def run_ours_box_or_points(image_np, pts_sampled, pts_labels, model):
    img_tensor = ToTensor()(image_np)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()


def process_folders(training_file_path, label_file_path, output_folder, append_to_csv=True):
    img_sticker = os.path.basename(training_file_path).split('.')[0]

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
            mask_efficient_sam_vitt = run_ours_box_or_points(rgb_image, input_point, input_label, efficient_sam_vitt_model)
            show_anns_ours(mask_efficient_sam_vitt, ax[1])
            iou_vitt = compute_iou(mask_efficient_sam_vitt, label_slice)
            dice_vitt = compute_dice(mask_efficient_sam_vitt, label_slice)
            ax[1].title.set_text(f"EfficientSAM (VIT-tiny)\nIoU: {iou_vitt:.4f}, Dice: {dice_vitt:.4f}")
            ax[1].axis('off')

            ax[2].imshow(rgb_image)
            mask_efficient_sam_vits = run_ours_box_or_points(rgb_image, input_point, input_label, efficient_sam_vits_model)
            show_anns_ours(mask_efficient_sam_vits, ax[2])
            iou_vits = compute_iou(mask_efficient_sam_vits, label_slice)
            dice_vits = compute_dice(mask_efficient_sam_vits, label_slice)
            ax[2].title.set_text(f"EfficientSAM (VIT-small)\nIoU: {iou_vits:.4f}, Dice: {dice_vits:.4f}")
            ax[2].axis('off')

            ax[3].imshow(label_slice, cmap='gray')
            ax[3].title.set_text(f'Label Slice {slice_index}')
            ax[3].axis('off')

            slice_filename = os.path.join(output_folder, f"{img_sticker}_segmented_slice_{slice_index}.png")
            plt.savefig(slice_filename)
            plt.close(fig)

            iou_dice_data.append({
                "Image Sticker": img_sticker,
                "Slice Index": slice_index,
                "Model": "EfficientSAM (VIT-tiny)",
                "IoU": iou_vitt,
                "Dice": dice_vitt
            })
            iou_dice_data.append({
                "Image Sticker": img_sticker,
                "Slice Index": slice_index,
                "Model": "EfficientSAM (VIT-small)",
                "IoU": iou_vits,
                "Dice": dice_vits
            })

    iou_dice_df = pd.DataFrame(iou_dice_data)
    csv_path = os.path.join(output_folder, "iou_dice_scores.csv")
    if append_to_csv and os.path.exists(csv_path):
        existing_iou_dice_df = pd.read_csv(csv_path)
        iou_dice_df = pd.concat([existing_iou_dice_df, iou_dice_df], ignore_index=True)
    iou_dice_df.to_csv(csv_path, index=False)


def main():
    training_data_folder = 'dataset/Task06_Lung/imagesTr'
    label_data_folder = 'dataset/Task06_Lung/labelsTr'
    output_folder = 'dataset/Task06_Lung/output'

    for filename in os.listdir(training_data_folder):
        if filename.endswith('.nii.gz') and not filename.startswith('._'):
            training_file_path = os.path.join(training_data_folder, filename)
            label_file_path = os.path.join(label_data_folder, filename)
            process_folders(training_file_path, label_file_path, output_folder)
        else:
            print(f'One of the paths does not exist: {training_data_folder}, {label_data_folder}')


if __name__ == "__main__":
    main()