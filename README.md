# DSFDcd:Joint Distribution Sampling and Feature Decoupling Deep Network for Remote Sensing Change Detection
Here, we provide the pytorch implementation of the paper: DSFDcd:Joint Distribution Sampling and Feature Decoupling Deep Network for Remote Sensing Change Detection

![](https://pic1.imgdb.cn/item/693fceb64a4e4213d006c607.png)

## Requirements

```
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
```

## Installation

Clone this repo:

```shell
git clone https://github.com/iceking111/DSFDcd.git
cd models
```



## Train

```python
python trainer.py
```



## Evaluate

```python
python evaluator.py
```


## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
└─label
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;


### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

DSIFN-CD: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Citation

If you use this code for your research, please cite our paper:

```
@ARTICLE{11063309,
  author={Wang, Bin and Jiang, Xiaohu and Qin, Pinle and Zeng, Jianchao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DSFDcd: Joint Distribution Sampling and Feature Decoupling Deep Network for Remote Sensing Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-12},
  keywords={Feature extraction;Prototypes;Training;Computational modeling;Transformers;Semantics;Data models;Generators;Data mining;Vegetation mapping;Change detection;distribution sampling;exceed-expectation (EE) loss;feature decoupling},
  doi={10.1109/TGRS.2025.3585229}}

```

