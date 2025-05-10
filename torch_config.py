import torch
import os

DATASET_PATH = "C:\\Users\\NIK\\DS_TASKS\\cells\\only_cells\\slicing\\"  # 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\BrightField'

IMAGE_PATH = os.path.join(DATASET_PATH, 'original')  # 'bright_field_0_png_norm_pp'
MASK_PATH = os.path.join(DATASET_PATH, 'mask')  # masks_unclear_0_pp

TEST_SPLIT = 0.20

BATCH_SIZE = 2  # 32
NUM_EPOCHS = 500  # 300
EPOCHS_NO_IMPROVE = 5

SEED_COUNTER = 1
IS_GRIDSEARCH = False

CNN_DEPTH = [4]  # [2, 3, 4, 5, 6, 7, 8, 9]
FIRST_CHANNELS = [64]  # [32, 64]
BATCH_NORM = [True]
DROP_OUT = [True]
KERNEL_SIZE = [3]

OPTIMIZER = 'Adam'  # SGD, RMSprop
AMSGRAD = [False]  # [False, True]
LR = [1e-3]  # [0.0024436950146627568]  # [7e-4]  # [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
MOMENT = [(0.9, 0.999)]  # [(0.9766666666666667, 0.9954566929133858)]  # [(0.9, 0.999)]  # [(0.9, 0.99), (0.9, 0.999), (0.9, 0.9999)]
WEIGHT_DECAY = [0]  # [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

IS_SAVE = True
BASE_OUTPUT = 'Bright_data/output/exp_1'
MODEL_NAME = 'unet_Adam_bright_data_1_exp_1.pth'# 'unet_Adam_bright_data_0.pth'
PATH_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT)
MODEL_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT, MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

