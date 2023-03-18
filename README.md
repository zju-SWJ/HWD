# Holistic Weighted Distillation for Semantic Segmentation

This repository contains the source code of HWD. You can download the pre-trained teacher models and ImageNet pre-trained student models from the URLs provided by [CIRKD](https://github.com/winycg/CIRKD). 

Since ADE20K and COCO-Stuff teacher models are not released when we submitted our paper, we trained these models by ourselves. Larger iterations (e.g., 80K, 160K) should be used when training the teacher.

## Paper
[https://zju-swj.github.io/papers/ICME2023-HWD/paper.pdf](https://zju-swj.github.io/papers/ICME2023-HWD/paper.pdf)

## General parameter choices
**model**  psp, psp_mobile, deeplabv3, deeplabv3_mobile 

**backbone** resnet18, resnet50, resnet101, mobile

## Hyper-parameters in HWD
Check the comments in train_hwd.py.

## Training Guidances

### First, modify the data path in settings.py!!!

### Train the baseline method (Cityscapes, DeepLabV3-ResNet18)
```
python -m torch.distributed.launch --nproc_per_node=4 train_baseline.py \
    --gpu-id 0,1,2,3 --dataset citys \
    --model deeplabv3 --backbone resnet18 \
    --pretrained-base [your checkpoint path]/resnet18-imagenet.pth
```

### Train the distillation method xxx (Cityscapes, DeepLabV3-ResNet101->DeepLabV3-ResNet18)
```
python -m torch.distributed.launch --nproc_per_node=4 train_xxx.py \
    --gpu-id 0,1,2,3 --dataset citys \
    --teacher-model deeplabv3 --teacher-backbone resnet101 \
    --teacher-pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-model deeplabv3 --student-backbone resnet18 \
    --student-pretrained-base [your checkpoint path]/resnet18-imagenet.pth
```

### Train HWD (Cityscapes, DeepLabV3-ResNet101->DeepLabV3-ResNet18)
```
python -m torch.distributed.launch --nproc_per_node=4 train_hwd.py \
    --gpu-id 0,1,2,3 --dataset citys \
    --teacher-model deeplabv3 --teacher-backbone resnet101 \
    --teacher-pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-model deeplabv3 --student-backbone resnet18 \
    --student-pretrained-base [your checkpoint path]/resnet18-imagenet.pth \
    --activation --L MSE --process E --parameter 0.1 --shift 1.0 --DivPixel N
```

## Evaluation Guidances

### First, modify the data path in settings.py!!!

### Eval the method xxx (Cityscapes, DeepLabV3-ResNet18)
```
python -m torch.distributed.launch --nproc_per_node=4 eval.py \
    --gpu-id 0,1,2,3 --dataset citys \
    --model deeplabv3 --backbone resnet18 \ 
    --method xxx \
    --pretrained [your checkpoint path]/xxx_deeplabv3_resnet18_citys_best_model.pth
```

## Citation
If you find this repository useful, please consider citing the following paper:
```
@inproceedings{sun2022holistic,
  title={Holistic Weighted Distillation for Semantic Segmentation},
  author={Sun, Wujie and Chen, Defang and Wang, Can and Ye, Deshi and Feng, Yan and Chen, Chun},
  year={2023},
  organization={IEEE},
}
```
