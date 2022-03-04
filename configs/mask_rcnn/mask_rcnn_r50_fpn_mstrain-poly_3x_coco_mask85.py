_base_ = [
    '../common/mstrain-poly_3x_sunrgbd_in_coco_style_instance_mask85.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]
# Seems that the pretrained performs poorly, let's use load from insted                 
load_from= 'checkpoints/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'
