"""
Mask image with a probability of p
"""
import cv2
import numpy as np
import os, glob


def mask_an_image(input_img_fn, output_img_path, mask_prob=0):
    """
    mask an image with probablity of mask_prob
    """
    fn = input_img_fn.split("/")[-1]
    img = cv2.imread(input_img_fn)
    # print(np.max(img))
    # print(np.min(img))
    H, W = img.shape[:2]
    mask = np.random.rand(H, W)
    mask = mask <= mask_prob
    img[mask] = 0
    cv2.imwrite(output_img_path + fn, img)


def process_foler(mask_prob):
    print(mask_prob)
    input_img_path = (
        "/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/val/dhs/"
    )
    img_files = sorted(glob.glob(input_img_path + "*.png"))
    print(len(img_files))
    output_img_path = input_img_path[:-1] + str(int(mask_prob * 100)) + "/"
    os.makedirs(output_img_path, exist_ok=True)
    print(output_img_path)
    for i, img_fn in enumerate(img_files):
        if i % 200 == 0:
            print("Processing ", i, "   ...")
        mask_an_image(img_fn, output_img_path, mask_prob)


if __name__ == "__main__":
    for mask_prob in [0.8, 0.6, 0.4, 0.2]:
        process_foler(mask_prob)
