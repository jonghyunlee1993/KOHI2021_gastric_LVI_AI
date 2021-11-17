import os
import cv2
import glob
import numpy as np
import pandas as pd

from config import *
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import train_test_split


def generate_patch_df(flist, label):
    df = pd.DataFrame({"fpath": flist})
    df['slide_id'] = df['fpath'].map(lambda x: x.split("/")[-1].split(".")[0].split("_")[0])
    df['patient_id'] = df['slide_id'].map(lambda x: x.split("-")[0])
    df['target'] = label

    df = df.loc[:, ["patient_id", "slide_id", "fpath", "target"]]
    
    return df


def define_dataset(positive_df, negative_df, normal_df, sampling_rate=0.2):
    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(positive_df['fpath'], positive_df['target'], test_size=sampling_rate, random_state=1234)
    X_train_pos, X_valid_pos, y_train_pos, y_valid_pos = train_test_split(X_train_pos, y_train_pos, test_size=sampling_rate, random_state=1234)

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(negative_df['fpath'], negative_df['target'], test_size=sampling_rate, random_state=1234)
    X_train_neg, X_valid_neg, y_train_neg, y_valid_neg = train_test_split(X_train_neg, y_train_neg, test_size=sampling_rate, random_state=1234)
    
    X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(normal_df['fpath'], normal_df['target'], test_size=sampling_rate, random_state=1234)
    X_train_normal, X_valid_normal, y_train_normal, y_valid_normal = train_test_split(X_train_normal, y_train_normal, test_size=sampling_rate, random_state=1234)
    
    X_train = np.hstack([X_train_pos, X_train_neg, X_train_normal])
    X_valid = np.hstack([X_valid_pos, X_valid_neg, X_valid_normal])
    X_test = np.hstack([X_test_pos, X_test_neg, X_test_normal])

    y_train = np.hstack([y_train_pos, y_train_neg, y_train_normal])
    y_valid = np.hstack([y_valid_pos, y_valid_neg, y_valid_normal])
    y_test = np.hstack([y_test_pos, y_test_neg, y_test_normal])
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def define_dataset_pos_back(positive_df, background_df, sampling_rate=0.2):
    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(positive_df['fpath'], positive_df['target'], test_size=sampling_rate, random_state=1234)
    X_train_pos, X_valid_pos, y_train_pos, y_valid_pos = train_test_split(X_train_pos, y_train_pos, test_size=sampling_rate, random_state=1234)

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(background_df['fpath'], background_df['target'], test_size=sampling_rate, random_state=1234)
    X_train_neg, X_valid_neg, y_train_neg, y_valid_neg = train_test_split(X_train_neg, y_train_neg, test_size=sampling_rate, random_state=1234)
    
    X_train = np.hstack([X_train_pos, X_train_neg])
    X_valid = np.hstack([X_valid_pos, X_valid_neg])
    X_test = np.hstack([X_test_pos, X_test_neg])

    y_train = np.hstack([y_train_pos, y_train_neg])
    y_valid = np.hstack([y_valid_pos, y_valid_neg])
    y_test = np.hstack([y_test_pos, y_test_neg])
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def define_augmentation():
    train_transforms = A.Compose([ 
        A.RandomCrop(width=IM_WIDTH, height=IM_HEIGHT, p=1.0),
        
        A.OneOf([
            A.Transpose(),
            A.HorizontalFlip(),
            A.VerticalFlip()
        ], p=0.5),

        A.OneOf([
            A.Blur(),
            A.GaussianBlur(),
            A.GaussNoise(),
            A.MedianBlur()
        ], p=0.2),

        A.OneOf([
            A.ChannelShuffle(),
            A.ColorJitter(),
            A.HueSaturationValue(),
            A.RandomBrightnessContrast()
        ], p=0.5),
        
        A.Normalize(p=1.0),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([ 
        A.Resize(width=IM_WIDTH, height=IM_HEIGHT, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2()
    ])

    return train_transforms, valid_transforms


class LVIDataset(Dataset):
    def __init__(self, X, y, transforms):
        self.X = X
        self.y = y
        self.transforms = transforms
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        image  = cv2.imread(self.X[idx])
        target = self.y[idx]

        augmented = self.transforms(image=image)
        image = augmented['image']
        
        return image, target