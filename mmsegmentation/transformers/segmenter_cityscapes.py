#!/usr/bin/env python
# coding: utf-8



# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)


# Let's take a look at the dataset
import mmcv
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np
from PIL import Image
# convert dataset annotation to semantic segmentation map
data_root = 'Cityscapes'
img_dir = 'images/train'
ann_dir = 'labels/train'
# define class and plaette for better visualization
classes = ('background', 'road', 'sidewalk', 'building','wall', 'fence', 'pole', 'traffic light', 'traffic sign',
          'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus','train','motorcycle','bicycle')
palette = [[0,0,0], [128, 64, 128], [244,35,232], [70,70,70], [102,102,156], [190,153,153], [153, 153, 153],
           [250, 170,30], [220, 220, 0], [107, 142, 35], [152,251,152], [70,130,180],[220,20,60],[255,0,0],
           [0,0,142], [0,0,70], [0,60,100], [0,80,100],[0,0,230],[119,11,32]]
print(len(classes))
print(len(palette))

'''
for file in mmcv.scandir(osp.join(data_root, ann_dir),suffix='labelIds.png'):
  seg_img = Image.open(osp.join(data_root, ann_dir, file)).convert('P')
  seg_img.putpalette(np.array(palette, dtype=np.uint8))
  seg_img.save(osp.join(data_root, ann_dir, file))
'''


# # Register the custom dataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class CityscapeDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, **kwargs):
    super().__init__(img_suffix='leftImg8bit.png', seg_map_suffix='gtFine_labelIds.png', 
                     **kwargs)
    assert osp.exists(self.img_dir)

    


# # Create a config file


from mmcv import Config
cfg = Config.fromfile('../configs/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py')


# In[9]:


from mmseg.apis import set_random_seed

#HYPERPARAMETERS TO TUNE : Learning rate, batch size, optimizer, dropout

#OTHER BACKBONES TO TRY: ViT-Base, ViT-Large, Swin Transformer

#THINGS TO TRY: other data augmentations, k-fold cross validation, Tensorboard, Pretrain on Cityscapes

#MAIN PROBLEM: class imbalance -> validation data has few to no examples of some classes -> val. accuracy ~ 0

# add CLASSES and PALETTE to checkpoint
cfg.checkpoint_config.meta = dict(
    CLASSES=classes,
    PALETTE=palette)

# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='LN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 20

# Modify dataset type and path
cfg.dataset_type = 'CityscapeDataset'
cfg.data_root = data_root

#Batch size
cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu= 8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (512, 1024)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = 'images/train'
cfg.data.train.ann_dir = 'labels/train'
cfg.data.train.pipeline = cfg.train_pipeline

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = 'images/val'
cfg.data.val.ann_dir = 'labels/val'
cfg.data.val.pipeline = cfg.test_pipeline

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = 'images/val'
cfg.data.test.ann_dir = 'labels/val'
cfg.data.test.pipeline = cfg.test_pipeline

cfg.load_from = 'checkpoints/vit_tiny_p16_384.pth'
#cfg.load_from = './saved_models/segmen/latest.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './saved_models/segmenter3'

cfg.runner.max_iters = 100
cfg.log_config.interval = 10
cfg.evaluation.interval = 50
cfg.checkpoint_config.interval = 50

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# # Train and Evaluation

# In[10]:


from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                meta=dict())


'''
# # Inference

# In[11]:


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

img = mmcv.imread('Cityscapes/images/train/aachen_000003_000019_leftImg8bit.png')

model.cfg = cfg
result = inference_segmentor(model, img)
plt.figure(figsize=(8, 6))
show_result_pyplot(model, img, result, palette)


# # Load a saved model for inference

# In[25]:


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import matplotlib.pyplot as plt

# Specify the path to model config and checkpoint file
checkpoint_file = './saved_models/segmenter1/latest.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(cfg, checkpoint_file, device='cuda:0')

img = mmcv.imread('KITTI/images/000008_10.png')
model.cfg = cfg
result = inference_segmentor(model, img)
plt.figure(figsize=(8, 6))
show_result_pyplot(model, img, result, palette)
'''

# In[ ]:




