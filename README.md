# Holistic Weighted Distillation for Semantic Segmentation

This repository contains the source code of HWD. You can download the pre-trained teacher models and ImageNet pre-trained student models from the URLs provided by [CIRKD](https://github.com/winycg/CIRKD). 

Since ADE20K and COCO-Stuff teacher models are not released when we submitted our paper, we trained these models by ourselves. Larger iterations (e.g., 80K, 160K) should be used when training the teacher.

## Paper (Non-final version, subject to copyright)
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
    --activation --L MSE --process E --parameter 0.1 --shift 1.0 --DivPixel None
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

## More Results
### Default Setting
Input Size 512x512, Batchsize 8, Iterations 40000, DeepLabV3-ResNet101->DeepLabV3-ResNet18

**bold** best result, *italic* worse than the baseline

### Logit Softmax in DIST
|Method|Cityscapes|ADE20K|
|---|:--:|:--:|
|Baseline         |71.784|29.862|
|CWD              |73.565|32.898|
|DIST (w/ Softmax) |73.42|30.168|
|DIST (w/o Softmax)|71.984|31.677|
|HWD              |**74.718**|**33.846**|

The DIST paper uses softmax for logit before calculating distillation loss, but in our experiments, the performance obtained with or without using softmax varies with the data set. Based on the experimental results in the table above, we did not use softmax for logit in our appendix experiments. Whether used or not, its performance did not surpass CWD.

### More Datasets with Default Distillation Iterations
|Method|Cityscapes (512x1024, Batchsize 16, Iterations 20000)|VOC|
|---|:--:|:--:|
|Baseline|72.793|71.979|
|SKD     |73.643|*71.277*|
|IFVD    |73.963|*70.907*|
|CWD     |**74.444**|**74.439**|
|CIRKD   |74.191|*71.275*|
|DIST    |73.019|72.575|
|MGD     |73.508|72.142|

### Longer Distillation Iterations (Iterations 80000)
|Method|Cityscapes|Cityscapes (512x1024, Batchsize 16, Iterations 40000)|VOC|ADE20K|
|---|:--:|:--:|:--:|:--:|
|Baseline|73.629|73.865|73.155|34.276|
|SKD     |74.1|75.548|74.021|34.875|
|IFVD    |74.799|76.265|74.087|*33.679*|
|CWD     |**75.942**|76.12|**75.809**|36.085|
|CIRKD   |74.944|**76.54**|74.772|*34.019*|
|DIST    |74.655|75.158|74.588|34.828|
|MGD     |74.419|75.512|74.591|35.259|
|HWD     |-|-|-|**37.243**|

It can be seen that although training with longer distillation iterations can help spatial-wise methods (SKD, IFVD, and CIRKD) achieve better performance than the baseline, these methods are still inferor to CWD in most cases. 

In addition, these methods seem to be more suitable for simple datasets with a small number of categories (Cityscapes with 19 categories and VOC with 21 categories) and can easily fall short of the baseline when the number of categories in the dataset increases dramatically (ADE20K with 150 categories). But the channel-wise methods (CWD and HWD) can cope with complex datasets well. Since our weight distribution is category-dependent, our approach is actually more applicable to complex datasets (e.g., ADE20K and COCO-Stuff).


## Citation
If you find this repository useful, please consider citing the following paper (will be updated once the paper is formally published):
```
@inproceedings{sun2023holistic,
  title={Holistic Weighted Distillation for Semantic Segmentation},
  author={Sun, Wujie and Chen, Defang and Wang, Can and Ye, Deshi and Feng, Yan and Chen, Chun},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2023},
  organization={IEEE},
}
```
