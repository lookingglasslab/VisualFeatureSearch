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

BATCH_SIZE = 256

class FeatureType(Enum):
    '''Enum of all features we work with in this project. 
    Can be used to describe a SavedFeature instance, 
    as well as a general utility.'''
    HOG = 1
    CNN = 2

def precompute(dataset_path, cache_path):
    if not torch.cuda.is_available():
        raise Exception('No GPU Available')

    gpu = torch.device('cuda:0')

    feature_model = models.vgg16(pretrained=True)
    feature_model = feature_model.features
    feature_model = feature_model.to(gpu)

    search_db = database.SearchSpaceDataset(dataset_path)
    dl = DataLoader(search_db, batch_size=BATCH_SIZE)

    store = zarr.DirectoryStore(cache_path)
    root = zarr.group(store=store, overwrite=True)
    out_feats = root.zeros(FeatureType.CNN,
            shape=(len(search_db), 512, 7, 7),
            chunks=(500, None, None, None))

    with torch.no_grad():
        it = iter(dl)
        for batch, idxs in it:
            batch = batch.to(gpu)
            features = feature_model(batch)
            features = features.cpu().numpy()
            idxs = idxs.numpy()
            out_feats[idxs[0]:idxs[0]+BATCH_SIZE] = features
            print('Progress:', idxs[0], '/', len(search_db))

def load_data(cache_path):
    store = zarr.DirectoryStore(cache_path)
    root = zarr.group(store=store, overwrite=False)
    return root[FeatureType.CNN]

def get_args():
    parser = argparse.ArgumentParser(description='Precompute the features for a given dataset')
    parser.add_argument('dataset_path', type=str, help='Path to input image dataset')
    parser.add_argument('cache_path', type=str, help='Path to store the resulting ZARR datastore')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    precompute(args.dataset_path, args.cache_path)