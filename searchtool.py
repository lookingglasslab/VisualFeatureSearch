from __future__ import annotations
import numpy as np

import torch
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset

def get_crop_rect(query_mask: np.ndarray, threshold=0) -> tuple[int]:
    '''crops a mask'''
    rows, cols = np.where(query_mask > threshold)
    top = np.min(rows)
    bot = np.max(rows) + 1
    left = np.min(cols)
    right = np.max(cols) + 1
    return top, left, bot, right

class SearchTool:
    def __init__(self, model, device):
        self._model = model.to(device)
        self._device = device

    def set_input_image(self, query_image: torch.Tensor):
        '''Assumes `query_image` is already preprocessed'''
        query_image = query_image.to(self._device)
        self._query_features = self._model(query_image[None, :, :, :])

    def compute(self, query_mask):
        raise NotImplementedError('Do not use the SearchTool base class')

    def compute_batch(self, query_mask: np.ndarray, batch_arr: np.ndarray | torch.Tensor) -> tuple[torch.Tensor]:
        top, left, bot, right = get_crop_rect(query_mask)
        cropped_mask = query_mask[top:bot, left:right]
        cropped_query_features = self._query_features[..., top:bot, left:right]

        mask_tensor = torch.tensor(cropped_mask).to(self._device)
        mask_tensor = mask_tensor[None, None, :, :] # reshape to match feature tensors

        region_query_features = cropped_query_features * mask_tensor
        norm_query_features = region_query_features \
            / torch.linalg.vector_norm(region_query_features, dim=[1, 2, 3], keepdim=True)

        q_height = bot - top
        q_width = right - left

        width = batch_arr.shape[3]
        height = batch_arr.shape[2]

        if isinstance(batch_arr, np.ndarray):
            batch_vecs = torch.from_numpy(batch_arr)
        else:
            batch_vecs = batch_arr
        
        batch_vecs = batch_vecs.to(self._device)
        batch_sims = torch.zeros(len(batch_vecs)).to(self._device)
        batch_xs = torch.zeros(len(batch_vecs)).to(self._device)
        batch_ys = torch.zeros(len(batch_vecs)).to(self._device)

        # CONVOLUTION IDEAS
        # goal is to find cos(theta) = A . B / (||A|| * ||B||)
        # - first do convolution between batch_vecs (tensor) and norm_query_features*mask_tensor (kernel)
        # - batch_vecs is not normalized, so we need to find vector mag. for each window we used
        #   - this can (maybe?) be done by first doing batch_vecs * batch_vecs (element-wise)
        #   - then, we can do a second convolution between squared vecs and the mask tensor to get squared magnitude
        #   - then just divide convolution outputs element-wise

        scaledSims = torch.conv2d(batch_vecs.double(), norm_query_features * mask_tensor)

        sq_batch_vecs = batch_vecs * batch_vecs
        sq_mask_tensor = mask_tensor * mask_tensor
        batch_mags = torch.conv2d(sq_batch_vecs.double().view(-1, 1, height, width), sq_mask_tensor)
        batch_mags = batch_mags.view(batch_vecs.shape[0], 
                                     batch_vecs.shape[1],
                                     height - q_height + 1,
                                     width - q_width + 1)
        batch_mags = torch.sum(batch_mags, 1, keepdim=True)
        batch_mags = torch.sqrt(batch_mags)

        window_sims = scaledSims / batch_mags
        window_sims = window_sims.view(window_sims.shape[0], -1)

        batch_sims, idxs = window_sims.max(dim=1)
        batch_xs = idxs % (width - q_width + 1)
        batch_ys = torch.div(idxs, width - q_width + 1, rounding_mode='floor')

        return batch_sims.cpu(), batch_xs.cpu(), batch_ys.cpu()

class LiveSearchTool(SearchTool):
    def __init__(self, model, device, dataset: Dataset, batch_size=64):
        super().__init__(model, device)
        self._dataset = dataset
        # get all feature vectors from dataset
        self._all_vecs = self.__get_feature_vecs(batch_size)

    def __get_feature_vecs(self, batch_size):
        dl = DataLoader(self._dataset, batch_size)
        with torch.no_grad():
            it = iter(dl)
            all_vecs = []
            for batch in it:
                batch = batch.to(self._device)
                all_vecs.append(self._model(batch).cpu())
                del batch
        return all_vecs
    
    def compute(self, query_mask):
        sims = []
        xs = []
        ys = []
        for batch in self._all_vecs:
            batch_sims, batch_xs, batch_ys = self.compute_batch(query_mask, batch)
            sims.append(batch_sims)
            xs.append(batch_xs)
            ys.append(batch_ys)
        return torch.cat(sims), torch.cat(xs), torch.cat(ys)

# idea: make a live batched version, where feature vecs are stored in Zarr but computed on the fly
# or: switch between RAM and GPU memory for batches

class CachedSearchTool(SearchTool):
    def __init__(self, model, dataset: Dataset, device):
        super().__init__(model, dataset, device)
