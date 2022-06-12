from searchtool import CachedSearchTool

from skimage.feature import hog as skhog
import torch
import zarr

# note: takes raw image input, not a tensor!
def hog_model(img):
    features = skhog(img,
                     visualize=False, 
                     feature_vector=False, 
                     cells_per_block=(3,3), 
                     pixels_per_cell=(8,8), 
                     multichannel=True)
    features = features.reshape((26, 26, -1)) # assuming input image is 224x224
    features = features.transpose((2, 0, 1))
    return torch.from_numpy(features)

class HOGSearchTool(CachedSearchTool):
    def __init__(self, cache: zarr.Array, device, batch_size=500):
        self._model = hog_model
        self._device = device
        self._cache = cache
        self._batch_size = batch_size

    def set_input_image(self, query_image: torch.Tensor):
        features = self._model(query_image)
        self._query_features = features.to(self._device)
