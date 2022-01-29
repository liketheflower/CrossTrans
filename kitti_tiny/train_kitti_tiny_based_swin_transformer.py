"""
This code has bunch of errors and can not run through
"""

import os.path as osp

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


from prepare_data.convert_data_format import KittiTinyDataset


def get_cfg(cfg_fn, checkpoint_fn):
    cfg = Config.fromfile(cfg_fn)

    # Modify dataset type and path
    cfg.dataset_type = "KittiTinyDataset"
    cfg.data_root = "kitti_tiny/"

    cfg.data.test.type = "KittiTinyDataset"
    cfg.data.test.data_root = "kitti_tiny/"
    cfg.data.test.ann_file = "train.txt"
    cfg.data.test.img_prefix = "training/image_2"

    cfg.data.train.type = "KittiTinyDataset"
    cfg.data.train.data_root = "kitti_tiny/"
    cfg.data.train.ann_file = "train.txt"
    cfg.data.train.img_prefix = "training/image_2"

    cfg.data.val.type = "KittiTinyDataset"
    cfg.data.val.data_root = "kitti_tiny/"
    cfg.data.val.ann_file = "val.txt"
    cfg.data.val.img_prefix = "training/image_2"

    # modify num classes of the model in box head
    # cfg.model.roi_head.bbox_head.num_classes = 3
    # cfg.model.roi_head.mask_head.num_classes = 3
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = checkpoint_fn

    # Set up working dir to save files and logs.
    cfg.work_dir = "./tutorial_exps"

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = "mAP"
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f"Config:\n{cfg.pretty_text}")
    return cfg


def train(cfg_fn, checkpoint_fn):
    cfg = get_cfg(cfg_fn, checkpoint_fn)
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == "__main__":
    # cfg_fn = "./../configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
    cfg_fn = "./../configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_no_mask.py"
    checkpoint_fn = "./../checkpoints/cascade_mask_rcnn_swin_tiny_patch4_window7.pth"
    train(cfg_fn, checkpoint_fn)
