# Test swin with intertrans
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from/epoch_100.pth
#save_dir="./vis_outputs/swin_with_intertrans/"
#python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir=${save_dir} > ./test_logs/test_swin_with_intertrans_at_epoch100.log
# un show used to test the inference speed
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ./test_logs_unshow/test_swin_with_intertrans_at_epoch100_unshow.log

# Test swin without intertrans
#cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain.py
#checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain/epoch_100.pth
#save_dir="./vis_outputs/swin_without_intertrans/"
#mkdir $save_dir
#python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir=${save_dir} > ./test_logs/test_swin_without_intertrans_at_epoch100.log



# Test resnet with intertrans
cfg=configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py                        
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/epoch_100.pth
save_dir="./vis_outputs/resnet_with_intertrans/"                                       
#mkdir $save_dir                                                                         
#python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir=${save_dir} > ./test_logs/test_resnet_with_intertrans_at_epoch100.log
# un show used to test the inference speed
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ./test_logs_unshow/test_resnet_with_intertrans_at_epoch100_unshow.log


# Test resnet without interTrans                                        
#cfg=configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain.py             
#checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain/epoch_100.pth
#save_dir="./vis_outputs/resnet_without_intertrans/"                                       
#mkdir $save_dir                                                                         
#python tools/test.py  ${cfg} ${checkpoint} --eval=bbox --show --show-dir=${save_dir} > ./test_logs/test_resnet_without_intertrans_at_epoch100.log
