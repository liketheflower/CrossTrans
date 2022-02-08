# Train with interTrans  train from scratch
#cfg=configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py
#python tools/train.py  ${cfg} > train_maskrcnn_with_intertrans_new_coco_dataset.log

# Train without interTrans  train from scratch
cfg= configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain.py 
python tools/train.py  ${cfg} > train_maskrcnn_with_intertrans_new_coco_dataset_no_pretrain.log
