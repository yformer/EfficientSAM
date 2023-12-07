# EfficientSAM
EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

## News
[Dec.6 2023] EfficientSAM demo is available on the [HuggingFace Space](https://huggingface.co/spaces/yunyangx/EfficientSAM) (huge thanks to all the HF team for their support).

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
Models for GPU/CPU are available at the file folder of [HuggingFace Space](https://huggingface.co/spaces/yunyangx/EfficientSAM/).

| EfficientSAM-S | EfficientSAM-Ti |
|------------------------------|------------------------------|
| [Download](https://www.dropbox.com/scl/fi/ziif8xudwbyyphb4tohza/efficientsam_s_gpu.jit?rlkey=8aflq9kf0bfujz5ex4lxuoq56&dl=0) |[Download](https://www.dropbox.com/scl/fi/lup5s4gthmlv6qf3f5zz3/efficientsam_ti_gpu.jit?rlkey=pap1xktxw50qiaey17no16bqz&dl=0)|

You can directly use EfficientSAM,
```
import torch

efficientsam = torch.jit.load(efficientsam_s_gpu.jit)
```

## Colab
The colab is shared [here](https://colab.research.google.com/drive/150dvh_lwbliC3020fWO9qASgy-so6sUZ?usp=sharing)



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
