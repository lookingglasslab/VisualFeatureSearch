from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import os

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

class SimpleDataset(Dataset):
    ''' A basic PyTorch dataset we use for some small-scale experiments.
        Works well with VFS, but is not required. '''
    def __init__(self, path, transform=net_transform, return_idxs=True):
        self._path = path
        self._all_images = sorted(os.listdir(path))
        self.transform = transform 
        self.return_idxs = return_idxs
    
    def __len__(self):
        return len(self._all_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self._path, self._all_images[idx])
        image = Image.open(img_path).convert('RGB')

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
