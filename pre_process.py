import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from EfficientSAM_onnx_example import predict_onnx


def convert_to_rgb(hu_image, min_hu=-1200, max_hu=3000):
    # normalize HU values to 0-255
    # as HU values range from -1000 to 1000
    hu_image_clipped = np.clip(hu_image, min_hu, max_hu)
    np_img_normalized = (hu_image_clipped - min_hu) / (max_hu - min_hu)*255
    np_img_normalized = np_img_normalized.astype(np.uint8)

    rgb_image = np.stack([np_img_normalized] * 3, axis=-1)
    return rgb_image

def process_folders(training_file_path, label_file_path, output_folder):
        img_index = os.path.basename(training_file_path).split('_')[1].split('.')[0]  # 'XXX' from 'lung_XXX.nii.gz'

        # load the training and label images
        training_img = nib.load(training_file_path)
        label_img = nib.load(label_file_path)
        batched_point_labels = np.array([[[1]]]).astype(np.float32)
        training_data = training_img.get_fdata()
        label_data = label_img.get_fdata()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # traverse through the slices
        for slice_index in range(label_data.shape[2]):
            label_slice = label_data[:, :, slice_index]

            # check if the slice has any positive pixels (target pixels)
            if np.any(label_slice > 0):
                training_slice = training_data[:, :, slice_index].T
                rgb_image = convert_to_rgb(training_slice, min_hu=-1200, max_hu=3000)
                transposed_image = np.transpose(rgb_image, (2, 0, 1))
                final_image = np.expand_dims(transposed_image, axis=0)
                final_image = final_image.astype(np.float32) / 255.0

                # call for select_erode function to select a pixel
                chosen_pixel = select_erode(label_slice)
                if chosen_pixel is None:
                    label_slice = (label_slice * 255).astype(np.uint8)  # Scale label
                    label_slice = np.stack([label_slice] * 3, axis=-1)
                    concatenated_array = np.concatenate((rgb_image, label_slice), axis=1)
                    concatenated_image = Image.fromarray(concatenated_array)
                    # Save the image
                    output_path = os.path.join(output_folder, f'concatenated_image_{img_index}_{slice_index}.png')
                    concatenated_image.save(output_path)
                    print(f'Image saved: {output_path}')
                    continue
                chosen_pixel_array = np.array([chosen_pixel])  # 1x2
                chosen_pixel_array = np.expand_dims(chosen_pixel_array, axis=0)  # 1x1x2
                chosen_pixel_array = np.expand_dims(chosen_pixel_array, axis=0) # 1x1x1x2
                chosen_pixel_array = chosen_pixel_array.astype(np.float32)

                predict_mask = predict_onnx(final_image, chosen_pixel_array, batched_point_labels)
                predict_mask = np.stack([predict_mask] * 3, axis=-1).astype(np.uint8) * 255  # Convert single channel to 3 channels

                label_image = (label_slice * 255).astype(np.uint8)  # Scale label
                label_image = np.stack([label_image] * 3, axis=-1)  # Convert single channel to 3 channels

                concatenated_array = np.concatenate((rgb_image, label_image, predict_mask), axis=1)
                concatenated_image = Image.fromarray(concatenated_array)

                # Save the image
                output_path = os.path.join(output_folder, f'concatenated_image_{img_index}_{slice_index}.png')
                concatenated_image.save(output_path)
                print(f'Image saved: {output_path}')



def select_erode(np_img, remaining_pixels=25):
    kernel = np.ones((5, 5), dtype=np.uint8)
    pixels_positive = np_img.sum()
    before_erosion = np_img.copy()
    while pixels_positive > remaining_pixels:
        after_erosion = cv2.erode(before_erosion, kernel, iterations=1)
        pixels_positive = after_erosion.sum()
        if pixels_positive == 0:
            break
        before_erosion = after_erosion

    # find the coordinates of the non-zero pixels
    nonzero_coords = np.argwhere(before_erosion > 0)

    # if there are non-zero pixels, randomly select one
    if nonzero_coords.size > 0:
        chosen_pixel = nonzero_coords[np.random.randint(0, len(nonzero_coords))]
        return tuple(chosen_pixel)
    else:
        # if there are no non-zero pixels, return None
        return None



training_data_folder= 'dataset/Task06_Lung/imagesTr'
label_data_folder= 'dataset/Task06_Lung/labelsTr'
output_folder = 'dataset/Task06_Lung/output'

for filename in os.listdir(training_data_folder):
    if filename.endswith('.nii.gz') and not filename.startswith('._'):
        training_file_path = os.path.join(training_data_folder, filename)
        label_file_path = os.path.join(label_data_folder, filename)
        process_folders(training_file_path, label_file_path, output_folder)
    else:
        print(f'One of the paths does not exist: {training_data_folder}, {label_data_folder}')
