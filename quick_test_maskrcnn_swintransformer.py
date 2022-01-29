from mmdet.apis import inference_detector, init_detector


# Test mask rcnn
# Choose to use a config and initialize the detector
config = "configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py"
# Setup a checkpoint file to load
checkpoint = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
# initialize the detector
model = init_detector(config, checkpoint, device="cuda:0")

img = "demo/demo.jpg"
result = inference_detector(model, img)
print("Result", result)


# Test swin transformer
# Choose to use a config and initialize the detector
config = "configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
# Setup a checkpoint file to load
checkpoint = "checkpoints/cascade_mask_rcnn_swin_tiny_patch4_window7.pth"
# initialize the detector
model = init_detector(config, checkpoint, device="cuda:0")

img = "demo/demo.jpg"
result = inference_detector(model, img)
print("Result of swin transformer", result)
