import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from PIL import Image
import numpy as np

import os

# some utility transforms for us to prepare our data
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L227
net_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # note to future self: cannot be used for masks
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

vis_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

class SearchSpaceDataset(Dataset):
    def __init__(self, path, transform=net_transform, return_idxs=True):
        self._path = path
        self._all_images = sorted(os.listdir(path))
        self.transform = net_transform
        self.return_idxs = return_idxs
    
    def __len__(self):
        return len(self._all_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self._path, self._all_images[idx])
        image = Image.open(img_path)

        # convert grayscale to color, if necessary
        color_image = Image.new('RGB', image.size)
        color_image.paste(image)
        image = color_image
            
        if self.transform:
            image = self.transform(image)
        if self.return_idxs:
            return image, idx
        else:
            return image
    
    # get the image at index idx as a PIL object
    def get_vis_image(self, idx):
        img_path = os.path.join(self._path, self._all_images[idx])
        image = Image.open(img_path)

        # use visual transform
        image = vis_transform(image)
        return image
