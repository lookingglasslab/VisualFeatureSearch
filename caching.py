import os
from enum import Enum
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms

import database
import zarr

def precompute(dataloader: DataLoader, model, cache_path, array_name):
    if not torch.cuda.is_available():
        raise Exception('No GPU Available')

    gpu = torch.device('cuda:0')
    # model = model.to(gpu)

    # get output dimensions
    img0 = dataloader.dataset[0]
    img0 = img0.to(gpu)
    output0 = model(img0[None, :, :, :])
    tmpfs = output0.shape
    feature_shape = (len(dataloader.dataset), tmpfs[1], tmpfs[2], tmpfs[3])

    # create caching store
    store = zarr.DirectoryStore(cache_path)
    root = zarr.group(store=store, overwrite=False)
    out_feats = root.zeros(array_name,
            shape=feature_shape,
            chunks=(500, None, None, None),
            overwrite=True)

    with torch.no_grad():
        it = iter(dataloader)
        idx = 0
        for batch in it:
            batch = batch.to(gpu)
            features = model(batch)
            features = features.cpu().numpy()
            length = min(dataloader.batch_size, len(out_feats) - idx)
            out_feats[idx:idx+length] = features[:length]
            idx += length
            print('Progress:', idx, '/', len(dataloader.dataset))
            del batch, features