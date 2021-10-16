import os
import cv2
import glob
import PIL
import openslide
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_binary_mask(slide_image, level_of_interest):
    slide_image_dimension = slide_image.level_dimensions[level_of_interest]
    slide_image = np.array(slide_image.read_region((0, 0), level_of_interest, (slide_image_dimension)).convert("RGB"))
    slide_image = cv2.cvtColor(slide_image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(slide_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary_image[binary_image > 0] = 255
    binary_image = 255 - binary_image

    binary_image = cv2.erode(binary_image, np.ones((5, 5), np.uint8), iterations=2)
    binary_image = cv2.dilate(binary_image, np.ones((7, 7), np.uint8), iterations=10)

    return binary_image


def save_binary_mask(binary_mask, patient_id, my_path):
    cv2.imwrite(os.path.join(my_path, f"{patient_id}_tissue_mask.png"), binary_mask)


if __name__ == "__main__":

    level_of_interest = 1
    my_path = "./data/LVI_dataset"
    WSI_flist = glob.glob(os.path.join(my_path, "*.svs"))

    print(f"WSI: {len(WSI_flist)}")

    for slide_image_path in tqdm(WSI_flist):
        try:
            patient_id = slide_image_path.split("/")[-1].split('.')[0]
            
            slide_image = openslide.OpenSlide(slide_image_path)
            binary_mask = generate_binary_mask(slide_image, level_of_interest)
            
            save_binary_mask(binary_mask, patient_id, my_path)
        except:
            print(f"Error: {patient_id}")