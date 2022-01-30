#cfg=./../configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_no_mask.py
cfg=./../configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
checkpoint=./../checkpoints/cascade_mask_rcnn_swin_tiny_patch4_window7.pth
python train_kitti_tiny_based_swin_transformer.py ${cfg}  --cfg-options model.pretrained=${checkpoint} model.backbone.use_checkpoint=True
