from __future__ import annotations
from enum import Enum

import os
import pickle

import numpy as np

# keep it located within the VisualSearch repo
CACHE_DIR = os.path.dirname(__file__) + '/SavedFeatures' if __file__ else './SavedFeatures'

class FeatureType(Enum):
    '''Enum of all features we work with in this project. 
    Can be used to describe a SavedFeature instance, 
    as well as a general utility.'''
    HOG = 1
    CNN = 2

class SavedFeature:
    '''Class used to save and load precomputed image features'''
    def __init__(self, path: str, pos: tuple[int, int], 
            size: tuple[int, int], data: np.ndarray, type: FeatureType = None):
        self.path = path
        self.pos = pos # x,y coordinates in the image
        self.size = size # width, height coordinates
        self.data = data
        self.datatype = type

    def __repr__(self):
        return 'SavedFeature(path=' + self.path + ', shape=' + str(self.data.shape) + ')'

def save_all(features: list[SavedFeature], filename: str):
    '''Save all given features to a file in "`CACHE_DIR`/`filename`".
    Uses pickle for serializing and saving.'''
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    path = os.path.join(CACHE_DIR, filename)
    
    with open(path, 'wb') as f:
        pickle.dump(features, f)

def load_from_file(filename: str, assemble_numpy=False):
    path = os.path.join(CACHE_DIR, filename)
    with open(path, 'rb') as f:
        featurelist: list[SavedFeature] = pickle.load(f)
    
    if assemble_numpy:
        all_vecs = [feat.data for feat in featurelist]
        return featurelist, np.stack(all_vecs)
    
    return featurelist