#ONNX export code is from [labelme annotation tool](https://github.com/labelmeai/efficient-sam). Huge thanks to Kentaro Wada.

import onnxruntime
import torch

from efficient_sam.build_efficient_sam import build_efficient_sam_vits
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

import onnx_models


def export_onnx(onnx_model, output, dynamic_axes, dummy_inputs, output_names):
    with open(output, "wb") as f:
        print(f"Exporting onnx model to {output}...")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    inference_session = onnxruntime.InferenceSession(output)
    output = inference_session.run(
        output_names=output_names,
        input_feed={k: v.numpy() for k, v in dummy_inputs.items()},
    )
    print(output_names)
    print([output_i.shape for output_i in output])


def export_onnx_esam(model, output):
    onnx_model = onnx_models.OnnxEfficientSam(model=model)
    dynamic_axes = {
        "batched_images": {0: "batch", 2: "height", 3: "width"},
        "batched_point_coords": {2: "num_points"},
        "batched_point_labels": {2: "num_points"},
    }
    dummy_inputs = {
        "batched_images": torch.randn(1, 3, 1080, 1920, dtype=torch.float),
        "batched_point_coords": torch.randint(
            low=0, high=1080, size=(1, 1, 5, 2), dtype=torch.float
        ),
        "batched_point_labels": torch.randint(
            low=0, high=4, size=(1, 1, 5), dtype=torch.float
        ),
    }
    output_names = ["output_masks", "iou_predictions"]
    export_onnx(
        onnx_model=onnx_model,
        output=output,
        dynamic_axes=dynamic_axes,
        dummy_inputs=dummy_inputs,
        output_names=output_names,
    )


def export_onnx_esam_encoder(model, output):
    onnx_model = onnx_models.OnnxEfficientSamEncoder(model=model)
    dynamic_axes = {
        "batched_images": {0: "batch", 2: "height", 3: "width"},
    }
    dummy_inputs = {
        "batched_images": torch.randn(1, 3, 1080, 1920, dtype=torch.float),
    }
    output_names = ["image_embeddings"]
    export_onnx(
        onnx_model=onnx_model,
        output=output,
        dynamic_axes=dynamic_axes,
        dummy_inputs=dummy_inputs,
        output_names=output_names,
    )


def export_onnx_esam_decoder(model, output):
    onnx_model = onnx_models.OnnxEfficientSamDecoder(model=model)
    dynamic_axes = {
        "image_embeddings": {0: "batch"},
        "batched_point_coords": {2: "num_points"},
        "batched_point_labels": {2: "num_points"},
    }
    dummy_inputs = {
        "image_embeddings": torch.randn(1, 256, 64, 64, dtype=torch.float),
        "batched_point_coords": torch.randint(
            low=0, high=1080, size=(1, 1, 5, 2), dtype=torch.float
        ),
        "batched_point_labels": torch.randint(
            low=0, high=4, size=(1, 1, 5), dtype=torch.float
        ),
        "orig_im_size": torch.tensor([1080, 1920], dtype=torch.long),
    }
    output_names = ["output_masks", "iou_predictions"]
    export_onnx(
        onnx_model=onnx_model,
        output=output,
        dynamic_axes=dynamic_axes,
        dummy_inputs=dummy_inputs,
        output_names=output_names,
    )


def main():
    # faster
    export_onnx_esam(
        model=build_efficient_sam_vitt(),
        output="weights/efficient_sam_vitt.onnx",
    )
    export_onnx_esam_encoder(
        model=build_efficient_sam_vitt(),
        output="weights/efficient_sam_vitt_encoder.onnx",
    )
    export_onnx_esam_decoder(
        model=build_efficient_sam_vitt(),
        output="weights/efficient_sam_vitt_decoder.onnx",
    )

    # more accurate
    export_onnx_esam(
        model=build_efficient_sam_vits(),
        output="weights/efficient_sam_vits.onnx",
    )
    export_onnx_esam_encoder(
        model=build_efficient_sam_vits(),
        output="weights/efficient_sam_vits_encoder.onnx",
    )
    export_onnx_esam_decoder(
        model=build_efficient_sam_vits(),
        output="weights/efficient_sam_vits_decoder.onnx",
    )


if __name__ == "__main__":
    main()
