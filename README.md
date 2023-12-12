# EfficientSAM
EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

## News
[Dec.11 2023] The model code is fully available in this repository. The [sample_inference](https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_example.py) binary shows how to instantiate the model and query points on an image.

[Dec.6 2023] EfficientSAM demo is available on the [Hugging Face Space](https://huggingface.co/spaces/yunyangx/EfficientSAM) (huge thanks to all the HF team for their support).

[Dec.5 2023] We release the torchscript version of EfficientSAM and share a colab.

## Online Demo & Examples
Online demo and examples can be found in the [project page](https://yformer.github.io/efficient-sam/).

## EfficientSAM Instance Segmentation Examples
  |   |   |
:-------------------------:|:-------------------------:
Point-prompt | ![point-prompt](figs/examples/demo_point.png)
Box-prompt |  ![box-prompt](figs/examples/demo_box.png)
Segment everything |![segment everything](figs/examples/demo_everything.png)
Saliency | ![Saliency](figs/examples/demo_saliency.png)

## Model
The weights files for EfficientSAM are available under the weights folder of this github repository. Example instantiations and run of the models can be found in [EfficientSAM_example.py](https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_example.py).

| EfficientSAM-S | EfficientSAM-Ti |
|------------------------------|------------------------------|
| [Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vits.pt.zip) |[Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vitt.pt)|

You can directly use EfficientSAM with checkpoints,
```
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
efficient_sam_vit = build_efficient_sam_vitt()
```

## Colab
The colab is shared [here](https://github.com/yformer/EfficientSAM/blob/main/notebooks/EfficientSAM_example.ipynb)



## Acknowledgement

+ [SAM](https://github.com/facebookresearch/segment-anything)
+ [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
+ [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
+ [U-2-Net](https://github.com/xuebinqin/U-2-Net)


If you're using EfficientSAM in your research or applications, please cite using this BibTeX:
```bibtex


@article{xiong2023efficientsam,
  title={EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything},
  author={Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra},
  journal={arXiv:2312.00863},
  year={2023}
}
```
