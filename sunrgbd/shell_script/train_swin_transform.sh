#cfg=configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
#cfg=configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_sunrgbd_in_coco_style.py
#checkpoint=checkpoints/cascade_mask_rcnn_swin_tiny_patch4_window7.pth  
#cfg=configs/swin//mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py
# Not use fp16
#checkpoint=checkpoints/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth

#python tools/train.py  ${cfg} --cfg-options model.pretrained=${checkpoint}



# Train with a pretrained model
# cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py
# python tools/train.py  ${cfg} > train_swin_NO_fp16.log

# Train WITHOUT a pretrained model, train from scratch
#cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain.py
#python tools/train.py  ${cfg} > train_swin_NO_fp16_no_pretrain.log


# Updated the coco setups using load from
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from.py 
python tools/train.py  ${cfg} > train_swin_load_from_new_coco_dataset.log

