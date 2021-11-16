import os

N_EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
PATIENCE = 20

IM_HEIGHT = 256
IM_WIDTH = 256

CLASSIFIACTION_MODE = "THREE_CLASSES"
MODEL_NAME = "resnet50"

PROJECT_PATH = "."
DATA_PATH = os.path.join(PROJECT_PATH, "data/LVI_dataset/")
DATA_POSITIVE_PATH = os.path.join(DATA_PATH, "patch_image_size-300_overlap-0/LVI/*.png")
DATA_NEGATIVE_PATH = os.path.join(DATA_PATH, "patch_image_size-300_overlap-0/Negative/*.png")
DATA_NORMAL_PATH = os.path.join(DATA_PATH, "patch_image_size-300_overlap-0/Normal/*.png")

CKPT_PATH = os.path.join(PROJECT_PATH, "checkpoints")
FINAL_WEIGHTS_PATH = os.path.join(PROJECT_PATH, f"weights/{CLASSIFIACTION_MODE}_{MODEL_NAME}_patch-{IM_WIDTH}.pt")

DATA_POSITIVE_LABEL = 1
DATA_NEGATIVE_LABEL = 0
DATA_NORMAL_LABEL = 2

SAMPLING_RATE = 0.2
NUM_WORKERS = 12
USE_TPU = False