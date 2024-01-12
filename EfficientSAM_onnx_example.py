#Onnx export code is from [labelme annotation tool](https://github.com/labelmeai/efficient-sam). Huge thanks to Kentaro Wada.

import numpy as np
import torch

import imgviz
import onnxruntime
import time
from PIL import Image


def predict_onnx(input_image, input_points, input_labels):
    if 0:
        inference_session = onnxruntime.InferenceSession(
            "weights/efficient_sam_vitt.onnx"
        )
        (
            predicted_logits,
            predicted_iou,
            predicted_lowres_logits,
        ) = inference_session.run(
            output_names=None,
            input_feed={
                "batched_images": input_image,
                "batched_point_coords": input_points,
                "batched_point_labels": input_labels,
            },
        )
    else:
        inference_session = onnxruntime.InferenceSession(
            "weights/efficient_sam_vitt_encoder.onnx"
        )
        t_start = time.time()
        image_embeddings, = inference_session.run(
            output_names=None,
            input_feed={
                "batched_images": input_image,
            },
        )
        print("encoder time", time.time() - t_start)

        inference_session = onnxruntime.InferenceSession(
            "weights/efficient_sam_vitt_decoder.onnx"
        )
        t_start = time.time()
        (
            predicted_logits,
            predicted_iou,
            predicted_lowres_logits,
        ) = inference_session.run(
            output_names=None,
            input_feed={
                "image_embeddings": image_embeddings,
                "batched_point_coords": input_points,
                "batched_point_labels": input_labels,
                "orig_im_size": np.array(input_image.shape[2:], dtype=np.int64),
            },
        )
        print("decoder time", time.time() - t_start)
    mask = predicted_logits[0, 0, 0, :, :] >= 0
    imgviz.io.imsave(f"figs/examples/dogs_onnx_mask.png", mask)


def main():
    image = np.array(Image.open("figs/examples/dogs.jpg"))

    input_image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    # batch_size, num_queries, num_points, 2
    input_points = np.array([[[[580, 350], [650, 350]]]], dtype=np.float32)
    # batch_size, num_queries, num_points
    input_labels = np.array([[[1, 1]]], dtype=np.float32)

    predict_onnx(input_image, input_points, input_labels)


if __name__ == "__main__":
    main()
