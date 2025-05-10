import numpy as np
from skimage.io import imread
import os


def get_dice(y_true: np.array, y_pred: np.array, smooth: int = 1) -> float:
    y_true = np.float64(y_true)/255
    y_pred = np.float64(y_pred)/255

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice


def get_metrics(true_path: str, pred_path: str) -> float:
    true_names = os.listdir(true_path)
    pred_names = os.listdir(pred_path)

    total_dice = 0
    for i in range(len(true_names)):
        print('--- --- --- --- ')
        print(true_names[i])
        print(pred_names[i])
        true_img = imread(os.path.join(true_path, true_names[i]))
        pred_img = imread(os.path.join(pred_path, pred_names[i]))
        print(true_img.dtype)
        dice = get_dice(true_img, pred_img)
        print(f'Dice coef: {dice}')
        total_dice += dice
    print(f'Average Dice coef on test data: {total_dice/len(true_names)}')
    return total_dice/len(true_names)


if __name__ == '__main__':
    get_metrics(true_path='C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\bright_data_251223\\result_data\\test_trypan_mask',
                pred_path='C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\bright_data_251223\\result_data\\pred_test_trypan_res')
