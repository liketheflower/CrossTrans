from mmdet.datasets.builder import DATASETS 
from mmdet.datasets.coco import CocoDataset
from data_preparation.convert_sunrgbd_to_coco_style import sunrgbd80_classsnames_ids_list
sun_rgbd_classes = tuple([category for category, _ in sunrgbd80_classsnames_ids_list])
print("sun_rgbd_classes: ", sun_rgbd_classes)
"""
 COCO original CLASSESS
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',             
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',               
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',             
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',         
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',         
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',              
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',            
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',         
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',           
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',                   
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',         
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',                
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',              
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush') 

"""
@DATASETS.register_module()                                                             
class SunrgbdDataset(CocoDataset):
    CLASSES = sun_rgbd_classes                                                                           
