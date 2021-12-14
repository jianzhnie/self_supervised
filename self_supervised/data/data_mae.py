'''
Author: jianzhnie
Date: 2021-12-14 17:20:54
LastEditTime: 2021-12-14 17:32:45
LastEditors: jianzhnie
Description:

'''

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder


class MaskGenerator:
    def __init__(self, input_size=224, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_ratio = mask_ratio

        if not isinstance(input_size, tuple):
            input_size = to_2tuple(input_size)

        self.height, self.width = input_size
        self.token_counts = self.height * self.width
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.height, self.width))
        return mask


class MAETransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE,
                                scale=(0.67, 1.),
                                ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                        std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        self.mask_generator = MaskGenerator(
            input_size=config.input_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(
                    default_collate(
                        [batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = MAETransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = DistributedSampler(dataset,
                                 num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank(),
                                 shuffle=True)
    dataloader = DataLoader(dataset,
                            config.DATA.BATCH_SIZE,
                            sampler=sampler,
                            num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn)

    return dataloader


if __name__ == '__main__':
    mask_generator = MaskGenerator()
    mask = mask_generator()
    print(mask.shape)
