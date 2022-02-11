"""
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
Some frame works need to use the 2D masks, since our system does not need 2D mask as an output, we will use dummy mask to train the model (set the mask branch loss weight as 0).
"""
import os, glob, json
import mmcv
import numpy as np

info = {
    "description": "SUN RGBD coco styple DHS images",
    "url": "",
    "version": "1.0",
    "year": 2022,
    "contributor": "Sun RGBD dataset creators, computer vision lab, Hunter College, CUNY",
    "date_created": "2022/01/01",
}


sunrgbd80_classsnames_ids_list = [
    ["person", 1],
    ["table", 2],
    ["desk", 3],
    ["sofa_chair", 4],
    ["pillow", 5],
    ["box", 6],
    ["garbage_bin", 7],
    ["cabinet", 8],
    ["drawer", 9],
    ["shelf", 10],
    ["kitchen_counter", 11],
    ["kitchen_cabinet", 12],
    ["cpu", 13],
    ["bench", 14],
    ["file_cabinet", 15],
    ["whiteboard", 16],
    ["lamp", 17],
    ["endtable", 18],
    ["bookshelf", 19],
    ["coffee_table", 20],
    ["dresser", 21],
    ["paper", 22],
    ["printer", 23],
    ["monitor", 24],
    ["back_pack", 25],
    ["night_stand", 26],
    ["door", 27],
    ["picture", 28],
    ["ottoman", 29],
    ["stool", 30],
    ["outlet", 31],
    ["towel", 32],
    ["tray", 33],
    ["bag", 34],
    ["stove", 35],
    ["bathtub", 36],
    ["scanner", 37],
    ["cubby", 38],
    ["mirror", 39],
    ["bottle", 40],
    ["rack", 41],
    ["cup", 42],
    ["thermos", 43],
    ["island", 44],
    ["counter", 45],
    ["bowl", 46],
    ["plate", 47],
    ["organizer", 48],
    ["switch", 49],
    ["pen", 50],
    ["coffee_maker", 51],
    ["cart", 52],
    ["tv_stand", 53],
    ["poster", 54],
    ["soap_dispenser", 55],
    ["toy", 56],
    ["chair", 57],
    ["sofa", 58],
    ["plant", 59],
    ["bed", 60],
    ["dining_table", 61],
    ["toilet", 62],
    ["tv", 63],
    ["laptop", 64],
    ["mouse", 65],
    ["basket", 66],
    ["keyboard", 67],
    ["telephone", 68],
    ["microwave", 69],
    ["oven", 70],
    ["paper_towel_dispenser", 71],
    ["sink", 72],
    ["fridge", 73],
    ["book", 74],
    ["clock", 75],
    ["vase", 76],
    ["fire_extinguisher", 77],
    ["blinds", 78],
    ["podium", 79],
    ["others", 80],
]


def get_categories_info():
    """
    "categories": [
        {"supercategory": "person","id": 1,"name": "person"},
        {"supercategory": "vehicle","id": 2,"name": "bicycle"},
        {"supercategory": "vehicle","id": 3,"name": "car"},
        {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
        {"supercategory": "vehicle","id": 5,"name": "airplane"},
        ...
        {"supercategory": "indoor","id": 89,"name": "hair drier"},
        {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ]
    """
    categories = [
        {"supercategory": "sunrgbd", "id": idx, "name": cat}
        for cat, idx in sunrgbd80_classsnames_ids_list
    ]
    print(categories)
    return categories


def get_images_info(img_files):
    """
    The image file has the format of "007663_dhs.png". For the coco style
    [
     {
        "license": 4, # we don't need this one
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
     },
     ...
    ]
    """
    assert len(img_files) > 0, "Image files are empty"

    def extract_info_(img_fn):
        file_name = img_fn.split("/")[-1]
        img_id = int(file_name.split("_")[0])
        image = mmcv.imread(img_fn)
        height, width = image.shape[:2]
        # print("image height, width", height, width)
        return {"file_name": file_name, "height": height, "width": width, "id": img_id}

    ret = [extract_info_(img_fn) for img_fn in img_files]
    return ret


def get_annotations_info(img_files, label_folder):
    """
    the coco style annotation has the following format. The annotation contains all the
    objects from the images. The id is the annotation id. We set it as image_id * 100 +
    the object index within this image. Segmentation will use the same as the bbox to
    generate a FAKE image segmentation. In the coco style, segmentation list of vertices
      (x, y pixel positions). COCO bounding box format is
      [top left x position, top left y position, width, height].
    [
     {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67],
        "category_id": 18,
        "id": 1768
     },
     ...
    ]
    """
    assert len(img_files) > 0, "Image files are empty"

    def extract_label_info_(img_fn):
        file_name = img_fn.split("/")[-1]
        img_id = int(file_name.split("_")[0])
        bbox_fn = (
            label_folder
            + "sunrgbd80_bbox/"
            + str(img_id).zfill(6)
            + "_gt_unnormalized_xmin_y_min_w_hs_sunrgbd80.npy"
        )
        label_fn = (
            label_folder
            + "sunrgbd80_bbox/"
            + str(img_id).zfill(6)
            + "_gt_class_ids_sunrgbd80.npy"
        )
        bboxes, labels = np.load(bbox_fn), np.load(label_fn)
        if bboxes.size == 0:
            return []
        bboxes = bboxes.tolist()
        ret = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            category_id = int(labels[i])
            # add point x1 y1 x2 y2 x1 y1 to avoid the problem mentioned here
            # https://github.com/cocodataset/cocoapi/issues/139
            segmentation = [bbox + bbox[:2]]
            area = bbox[2] * bbox[3]
            anno_id = img_id * 100 + i
            ret.append(
                {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "id": anno_id,
                }
            )
        return ret

    ret = []
    for fn in img_files:
        ret += extract_label_info_(fn)
    return ret


def generate_annotation(img_folder, label_folder, annotation_filename):
    """
    COCO annotation json file has the following keys
    dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    we only need generate "images" and "annotations"
    The creation follows this article
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    """
    img_files = sorted(glob.glob(img_folder + "*.png"))
    print(f"len of img_files {len(img_files)}")
    images_info = get_images_info(img_files)
    annotations_info = get_annotations_info(img_files, label_folder)
    categories_info = get_categories_info()
    final_annotations = {
        "info": info,
        "licenses": "none",
        "images": images_info,
        "annotations": annotations_info,
        "categories": categories_info,
    }
    fn = label_folder + annotation_filename
    with open(fn, "w") as f:
        json.dump(final_annotations, f)


def convert_data_to_coco_style(dataset_folder, img_type="rgb"):
    """
    Convert our SUN RGBD dataset to coco style
    """
    # For train
    img_folder = dataset_folder + "train/" + img_type + "/"
    label_folder = dataset_folder + "train/gts/raw_gts/"
    generate_annotation(img_folder, label_folder, "det_train"+img_type + ".json")
    # For test
    img_folder = dataset_folder + "val/"+ img_type + "/"      
    label_folder = dataset_folder + "val/gts/raw_gts/"
    generate_annotation(img_folder, label_folder, "det_val"+img_type + ".json")


if __name__ == "__main__":
    sunrgbd_folder = "/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/"
    convert_data_to_coco_style(sunrgbd_folder, "rgb")
