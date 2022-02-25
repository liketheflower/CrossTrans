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
    print(np.max(img))
    print(np.min(img))
    H, W = img.shape[:2]
    mask = np.random.rand(H, W)
    mask = mask > mask_prob
    img[mask] = 0
    cv2.imwrite(output_img_path + fn, img)


if __name__ == "__main__":
    output_img_path = "./../../demo/outputs/"
    os.makedirs(output_img_path, exist_ok=True)
    input_img_fn = "./../../demo/006223_dhs.png"
    mask_an_image(input_img_fn, output_img_path, 0.6)
