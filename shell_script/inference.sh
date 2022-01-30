# single-gpu testing
#python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm
config=configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
checkpoint=cascade_mask_rcnn_swin_tiny_patch4_window7.pth
python tools/test.py ${config} ${checkpoint} --eval bbox segm

