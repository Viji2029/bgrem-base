import glob
import os

import cv2
import numpy as np
import PIL
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image



class SalObjDataset(Dataset):
    def __init__(self, base_path, mode, transform=None, sz=320, rc=288,n_samples=None):
        self.base_path = base_path
        self.mode = mode
        self.n_samples = n_samples
   
        self.sz = sz
        self.rc = rc
        self.base_names = [
            f.split("/")[-1].replace(".jpg", "") for f in glob.glob(os.path.join(self.base_path, "fg", "*.jpg"))
        ]
        print("len base names", len(self.base_names))

        if self.n_samples is not None:
            #print("\n"*3, self.n_samples)
            image_names = random.sample(self.base_names,self.n_samples)
            #print("\n"*3, len(image_names))
            #random_idx = random.sample(range(len(self.base_names)), n_samples)
            self.images = [os.path.join(self.base_path, "fg", f"{f}.jpg") for f in image_names]
            self.masks = [os.path.join(self.base_path, "alpha", f"{f}.png") for f in image_names]

        else:
            self.images = [os.path.join(self.base_path, "fg", f"{f}.jpg") for f in self.base_names]
            self.masks = [os.path.join(self.base_path, "alpha", f"{f}.png") for f in self.base_names]
        
       
        
        self.transform = transform

    def __len__(self):
        if self.n_samples is not None:
            #print(self.n_samples)
            return self.n_samples
        else:
            return len(self.base_names)

    def __getitem__(self, idx):
        if self.n_samples is not None:
            assert idx < self.n_samples
        image = Image.open(self.images[idx])
        image.thumbnail((self.sz, self.sz), Image.Resampling.LANCZOS)
        mask = Image.open(self.masks[idx]).convert("L")
        mask.thumbnail((self.sz, self.sz), Image.Resampling.LANCZOS)

        # Experimented with flip, didn't improve results
        # if self.mode == "train":
        #     # if np.random.random() < 0.5:
        #     #     # hflip
        #     #     image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        #     #     mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        #     if np.random.random() < 0.5:
        #         # vflip
        #         image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        #         mask = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        image = np.array(image).astype("float32")

        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]

        mask = np.array(mask).astype("float32")

        image /= 255.0
        image -= (0.5, 0.5, 0.5)
        image /= (0.5, 0.5, 0.5)
        mask /= 255.0

        mask = np.expand_dims(mask, -1)

        pad_x = int(self.sz - image.shape[0])
        pad_y = int(self.sz - image.shape[1])

        image = np.pad(image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant")
        mask = np.pad(mask, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant")

        if self.mode == "train":
            # random crop to rc
            top = np.random.randint(0, self.sz - self.rc)
            left = np.random.randint(0, self.sz - self.rc)
            image = image[top : top + self.rc, left : left + self.rc]
            mask = mask[top : top + self.rc, left : left + self.rc]

        # if self.mode == "train":
        #     flipped_image = np.fliplr(image)
        #     flipped_mask = np.fliplr(mask)
        #     flipped_image = np.transpose(flipped_image, (2, 0, 1))
        #     flipped_mask = np.transpose(flipped_mask, (2, 0, 1))

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return torch.FloatTensor(image), torch.FloatTensor(mask)
