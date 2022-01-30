# Paper name

## Data preparation  
the sun rgb d depth images are converted to point cloud and then further converted to 
DHS images. We have the following components
- DHS images
- 2D bboxes:
the 2D bbox is saved in 009290_gt_unnormalized_xmin_y_min_w_hs_sunrgbd80.txt, it has the pixel value of top left x and y, also the 
width and height information.
such as 
```bash
1.000000000000000000e+00 1.940000000000000000e+02 1.240000000000000000e+02 2.470000000000000000e+02
1.000000000000000000e+00 3.130000000000000000e+02 1.390000000000000000e+02 1.280000000000000000e+02
2.840000000000000000e+02 4.030000000000000000e+02 3.300000000000000000e+01 3.800000000000000000e+01
```
- 2D lables  
We have 80 categories and those labels are saved in such as "009290_gt_class_ids_sunrgbd80.txt" file.
```bash
2.800000000000000000e+01
1.700000000000000000e+01
5.700000000000000000e+01
```

Some frame works need to use the 2D masks, since our system does not need 2D mask as an output,
 we will use dummy mask to train the model (set the mask branch loss weight as 0).
Run the [data preparation code](.data_preparation/convert_sunrgbd_to_coco_style.py) to 
generate coco style annotations.

# Train 2D object detector based on FPN

# Train 2D object detector based on Swin Transformer

