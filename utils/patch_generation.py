import os
import cv2
import PIL
import glob
import json
import openslide
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread


PIL.Image.MAX_IMAGE_PIXELS = None


def generate_result_directory(result_path):
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
        
        positive_dir = os.path.join(result_path, "positive")
        negative_dir = os.path.join(result_path, "negative")

        os.mkdir(positive_dir)
        os.mkdir(negative_dir)


def read_slide_image(slide_image_path, level_of_interest):
    slide_image = openslide.OpenSlide(slide_image_path)
    slide_image_dimension = slide_image.level_dimensions[level_of_interest]
    original_dimension = slide_image.level_dimensions[0]

    return slide_image, slide_image_dimension, original_dimension


def read_tissue_mask(tissue_mask_path, slide_image_dimension, original_dimension):

    mask_image = cv2.imread(tissue_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, slide_image_dimension, interpolation=cv2.INTER_LINEAR)
    mask_image_origin_dimension = cv2.resize(mask_image, original_dimension, interpolation=cv2.INTER_LINEAR)

    return mask_image, mask_image_origin_dimension


def read_geojson(geojson_path):
    geojson_string = open(geojson_path)
    geojson = json.load(geojson_string)

    return geojson


def get_label_from_geojson(geojson, level_of_interest):
    class_names = []
    square_coords = []

    for annotation_item in geojson['features']:
        geometry = annotation_item['geometry']
        
        class_name = annotation_item['properties']['classification']['name']
        class_names.append(class_name)
        
        if geometry['type'] == 'Polygon':
            coordinates = geometry['coordinates'][0]

            square_area, coords = calc_area(coordinates)
            square_coords.append(coords)

    return class_names, square_coords


def calc_area(coordinates):    
    my_array = np.array(coordinates)
    x_max = np.max(my_array[:, 0])
    x_min = np.min(my_array[:, 0])
    y_max = np.max(my_array[:, 1])
    y_min = np.min(my_array[:, 1])

    square_area = (x_max - x_min) * (y_max - y_min)
    
    return square_area, list(map(lambda x: int(x), [x_min, y_min, x_max, y_max]))


def extract_patch_from_slie_image(patient_id, slide_image, tissue_mask, class_names, square_coords, patch_size, stride, patch_result_path, tissue_threshold=0.7):
    patch_size_up = patch_size * (2 ** level_of_interest)

    for i, box in enumerate(square_coords):
        # print(f"patient id: {patient_id} box number: {i} / {len(square_coords)}", sep="\r")
        for x in range(box[0] - int(patch_size_up / 2), box[2] - int(patch_size_up / 2), stride):
            for y in range(box[1] - int(patch_size_up / 2), box[3] - int(patch_size_up / 2), stride):
                patch_image_from_tissue_mask = tissue_mask[y:y+patch_size_up, x:x+patch_size_up]

                if np.mean(patch_image_from_tissue_mask) / 255 >= tissue_threshold:
                    patch_image = np.array(slide_image.read_region((x, y), 0, (patch_size_up, patch_size_up)).convert("RGB"))
                    is_positive = True if class_names[i] == "LVI" else False

                    save_patch_image(patient_id, patch_image, patch_size, is_positive, x, y, patch_result_path)
                    

def save_patch_image(patient_id, patch_image, patch_size, is_positive, x, y, patch_result_path):
    patch_image = cv2.resize(patch_image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    if is_positive:
        cv2.imwrite(os.path.join(patch_result_path, "positive", f"{patient_id}_LVI_patch_{x}-{y}.png"), patch_image)
    else:
        cv2.imwrite(os.path.join(patch_result_path, "negative", f"{patient_id}_non-LVI_patch_{x}-{y}.png"), patch_image)


if __name__ == "__main__":
    level_of_interest = 2
    patch_size = 400
    overlap = round(patch_size / 4)
    stride = patch_size - overlap
    
    data_path = "./data/LVI_dataset"
    patch_result_path = os.path.join(data_path, f"patch_image_size-{patch_size}_overlap-{overlap}")

    generate_result_directory(patch_result_path)
    flist = glob.glob(data_path + "/*.svs")

    for slide_image_path in tqdm(flist):   
        subject_id = slide_image_path.split("/")[-1].split(".")[0]
        # try:
        slide_image, slide_image_dimension, original_dimension = read_slide_image(slide_image_path, level_of_interest)

        tissue_mask_path = slide_image_path.replace(".svs", "_tissue_mask.png")
        tissue_mask, tissue_mask_original_dimension = read_tissue_mask(tissue_mask_path, slide_image_dimension, original_dimension)

        geojson_path = slide_image_path.replace(".svs", ".geojson")
        geojson = read_geojson(geojson_path)
        class_names, square_coords = get_label_from_geojson(geojson, level_of_interest)

        extract_patch_from_slie_image(subject_id, slide_image, tissue_mask_original_dimension, class_names, square_coords, patch_size, stride, patch_result_path, tissue_threshold=0.7)
        break
        # except:
        #     print(f"patch generation was not conducted: s{subject_id}")