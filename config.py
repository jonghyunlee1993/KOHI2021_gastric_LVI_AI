import os

N_EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
PATIENCE = 20

IM_HEIGHT = 256
IM_WIDTH = 256

CLASSIFICATION_MODE = "LVI_background"
MODEL_NAME = "efficientnet_b4"

PROJECT_PATH = "."
DATA_PATH = os.path.join(PROJECT_PATH, "data/LVI_dataset/")
DATA_POSITIVE_PATH = os.path.join(DATA_PATH, "patch_image_size-300_overlap-0/LVI/*.png")

DATA_NEGATIVE_PATH = os.path.join(DATA_PATH, "patch_image_size-300_overlap-0/Negative/*.png")
DATA_NORMAL_PATH = os.path.join(DATA_PATH, "patch_image_size-300_overlap-0/Normal/*.png")

CKPT_PATH = os.path.join(PROJECT_PATH, "weights", f"{CLASSIFICATION_MODE}-{MODEL_NAME}_checkpoints")

DATA_POSITIVE_LABEL = 1
DATA_NEGATIVE_LABEL = 0
DATA_NORMAL_LABEL = 2

DATA_BACKGROUND_LABEL = 0

SAMPLING_RATE = 0.2
NUM_WORKERS = 12
USE_TPU = False
