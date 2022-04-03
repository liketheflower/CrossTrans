
# Updated the coco setups using load from
epoch=50
#for mask_prob in 20 40 60 80
for mask_prob in 95 90 85
do
echo "mask prob is ${mask_prob}"
cfg=configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_mask${mask_prob}.py                        
python tools/train.py  ${cfg} > train_maskrcnn_with_intertrans_new_coco_dataset_mask${mask_prob}.log

epoch_path=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_mask${mask_prob}/
last_epoch_path=${epoch_path}last_epoch/
mkdir -p ${last_epoch_path}
mv ${epoch_path}epoch_${epoch}.pth  ${last_epoch_path}
mv ${epoch_path}latest.pth  ${last_epoch_path}
echo "remove checkpoint to save space"
rm ${epoch_path}*.pth
echo "resnet is done now let's train swin"

cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from_mask${mask_prob}.py 
python tools/train.py  ${cfg} > train_swin_load_from_new_coco_dataset_mask${mask_prob}.log
epoch_path=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from_mask${mask_prob}/
last_epoch_path=${epoch_path}last_epoch/
mkdir -p ${last_epoch_path}
mv ${epoch_path}epoch_${epoch}.pth  ${last_epoch_path}
mv ${epoch_path}latest.pth  ${last_epoch_path}
echo "remove checkpoint to save space"
rm ${epoch_path}*.pth
done
