import os
import cv2
import PIL
import glob
import json
import openslide
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from tqdm import tqdm
from skimage.io import imread


PIL.Image.MAX_IMAGE_PIXELS = None


def generate_result_directory(result_path):
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
        
        positive_dir = os.path.join(result_path, "LVI")
        negative_dir = os.path.join(result_path, "Negative")
        normal_dir = os.path.join(result_path, "Normal")

        os.mkdir(positive_dir)
        os.mkdir(negative_dir)
        os.mkdir(normal_dir)


def read_slide_image(slide_image_path, level_of_interest):
    slide_image = openslide.OpenSlide(slide_image_path)
    slide_image_dimension = slide_image.level_dimensions[0]
    level_of_interest_dimension = slide_image.level_dimensions[level_of_interest]

    return slide_image, slide_image_dimension, level_of_interest_dimension


def read_tissue_mask(tissue_mask_path, slide_image_dimension):

    mask_image = cv2.imread(tissue_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, slide_image_dimension, interpolation=cv2.INTER_LINEAR)

    return mask_image


def read_geojson(geojson_path):
    geojson_string = open(geojson_path)
    geojson = json.load(geojson_string)

    return geojson


def get_label_from_geojson(geojson, level_of_interest):
    class_names = []
    square_coords = []

    for annotation_item in geojson['features']:
        geometry = annotation_item['geometry']

        try:
            class_name = annotation_item['properties']['classification']['name']
            class_names.append(class_name)
        except:
            pass

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


def extract_patch_from_slie_image(patient_id, slide_image, slide_image_dimension, tissue_mask, class_names, square_coords, patch_size, stride, patch_result_path, tissue_threshold=0.7):
    patch_size_in_origin_dimension = patch_size * (2 ** level_of_interest)
    positive_mask = np.zeros(slide_image_dimension, bool)

    # generate positive patch
    for i, box in enumerate(square_coords):
        try:
            if class_names[i] == "LVI":
                for x in range(box[0] - int(patch_size_in_origin_dimension / 2), box[2] + int(patch_size_in_origin_dimension / 2), stride):
                    if x + patch_size_in_origin_dimension > box[2] + int(patch_size_in_origin_dimension / 2):
                        break

                    for y in range(box[1] - int(patch_size_in_origin_dimension / 2), box[3] + int(patch_size_in_origin_dimension / 2), stride): 
                        if y + patch_size_in_origin_dimension > box[3] + int(patch_size_in_origin_dimension / 2):
                            break

                        positive_mask[y:y+patch_size_in_origin_dimension, x:x+patch_size_in_origin_dimension] = True
                        patch_image = np.array(slide_image.read_region((x, y), 0, (patch_size_in_origin_dimension, patch_size_in_origin_dimension)).convert("RGB"))
                        save_patch_image(patient_id, patch_image, patch_size, x, y, patch_result_path, class_names[i])

            elif class_names[i] in ("Negative", "Normal"):
                for x in range(0, slide_image_dimension[1], patch_size_in_origin_dimension):
                    for y in range(0, slide_image_dimension[0], patch_size_in_origin_dimension):
                        patch_positive_mask = positive_mask[y:y + patch_size_in_origin_dimension, x:x + patch_size_in_origin_dimension]
                        patch_tissue_mask = tissue_mask[y:y + patch_size_in_origin_dimension, x:x + patch_size_in_origin_dimension]

                        if patch_positive_mask.sum() == 0 and patch_tissue_mask.mean() / 255 >= tissue_threshold:
                            if class_names[i] == "Normal":
                                patch_image = np.array(slide_image.read_region((x, y), 0, (patch_size_in_origin_dimension, patch_size_in_origin_dimension)).convert("RGB"))
                                save_patch_image(patient_id, patch_image, patch_size, x, y, patch_result_path, class_names[i])
                            else:
                                patch_image = np.array(slide_image.read_region((x, y), 0, (patch_size_in_origin_dimension, patch_size_in_origin_dimension)).convert("RGB"))
                                save_patch_image(patient_id, patch_image, patch_size, x, y, patch_result_path, class_names[i])
        except:
            print(f"Label Error: {i}")
                    

def save_patch_image(patient_id, patch_image, patch_size, x, y, patch_result_path, class_name):
    patch_image = cv2.resize(patch_image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(patch_result_path, class_name, f"{patient_id}_{class_name}_patch_{x}-{y}.png"), patch_image)


if __name__ == "__main__":
    level_of_interest = 2
    patch_size = 300
    # overlap = round(patch_size / 5)
    overlap = 0
    stride = patch_size - overlap
    
    data_path = "./data/LVI_dataset"
    patch_result_path = os.path.join(data_path, f"patch_image_size-{patch_size}_overlap-{overlap}")

    generate_result_directory(patch_result_path)
    flist = glob.glob(os.path.join(data_path, "svs", "*.svs"))

    for slide_image_path in tqdm(flist):   
        patient_id = slide_image_path.split("/")[-1].split(".")[0]

        slide_image, slide_image_dimension, level_of_interest_dimension = read_slide_image(slide_image_path, level_of_interest)

        tissue_mask_path = slide_image_path.replace("/svs/", "/tissue_mask/").replace(".svs", "_tissue_mask.png")
        tissue_mask = read_tissue_mask(tissue_mask_path, slide_image_dimension)

        geojson_path = slide_image_path.replace("/svs/", "/geojson/").replace(".svs", ".geojson")
        geojson = read_geojson(geojson_path)
        class_names, square_coords = get_label_from_geojson(geojson, level_of_interest)
        
        extract_patch_from_slie_image(patient_id, slide_image, slide_image_dimension, tissue_mask, class_names, square_coords, patch_size, stride, patch_result_path, tissue_threshold=0.7)
