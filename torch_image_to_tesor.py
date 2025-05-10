import numpy as np
from skimage import io, img_as_float64
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imutils import paths
import os
import random
import torch_config
from sklearn.model_selection import train_test_split


class SegmentData(Dataset):
    def __init__(self, image_paths, mask_paths, trans=None, trans_masks=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.trans = trans
        self.trans_masks = trans_masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        mask_path = self.mask_paths[item]

        image = io.imread(image_path)
        mask = io.imread(mask_path)

        if self.trans is not None:
            image = self.trans(image)

        if self.trans_masks is not None:
            mask = self.trans_masks(mask)

        return (image, mask)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**19
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data():
    # torch.manual_seed(19)
    # torch.cuda.manual_seed_all(19)
    # random.seed(19)
    # np.random.seed(19)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(19)

    image_paths = sorted(list(paths.list_images(torch_config.IMAGE_PATH)))
    mask_paths = sorted(list(paths.list_images(torch_config.MASK_PATH)))
    split = train_test_split(image_paths, mask_paths, test_size=torch_config.TEST_SPLIT, random_state=19)
    (train_images, test_images) = split[:2]
    (train_masks, test_masks) = split[2:]
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
                                #  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)])
    trans_masks = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    train_data = SegmentData(train_images, train_masks, trans, trans_masks)
    test_data = SegmentData(test_images, test_masks, trans, trans_masks)
    # print(f'{len(train_data)} examples for train')
    # print(f'{len(test_data)} examples for test')

    g = torch.Generator()
    g.manual_seed(19)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=torch_config.BATCH_SIZE,
                              pin_memory=torch_config.PIN_MEMORY, num_workers=os.cpu_count(),
                              worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=torch_config.BATCH_SIZE,
                             pin_memory=torch_config.PIN_MEMORY, num_workers=os.cpu_count(),
                             worker_init_fn=seed_worker, generator=g)  # os.cpu_count()

    return train_loader, test_loader, len(train_data), len(test_data)


if __name__ == '__main__':
    # x_p = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Images_train'
    # y_p = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\Masks_train'
    # img_to_array(x_path=x_p, y_path=y_p)
    # what = paths.list_images(torch_config.IMAGE_PATH)
    tn, ts, _, __ = get_data()
