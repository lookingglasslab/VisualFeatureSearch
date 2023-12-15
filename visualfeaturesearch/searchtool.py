from __future__ import annotations
import numpy as np
import zarr

import torch
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
    '''Base class for searching across feature tensors. `compute()` is not implemented here, so
       it is recommended to use `CachedSearchTool` or `LiveSearchTool` instead.'''

    def __init__(self, model, device):
        self._model = model
        self._device = device

    def set_input_image(self, query_image: torch.Tensor):
        '''Assumes `query_image` is already preprocessed'''
        query_image = query_image.to(device=self._device, dtype=torch.float32)
        self._query_features = self._model(query_image[None, :, :, :]).to(self._device)

    def compute(self, query_mask):
        raise NotImplementedError('Do not use the SearchTool base class')

    def compute_batch(self, query_mask: np.ndarray, batch_arr: np.ndarray | torch.Tensor) -> tuple[torch.Tensor]:
        '''Computes cosine similarities for a single batch.'''
        top, left, bot, right = get_crop_rect(query_mask)
        cropped_mask = query_mask[top:bot, left:right]
        cropped_query_features = self._query_features[..., top:bot, left:right]

        # TODO: doing this once per batch is a potential bottleneck -- switch to doing it once
        mask_tensor = torch.tensor(cropped_mask, dtype=torch.float32).to(self._device)
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
        
        batch_vecs = batch_vecs.to(device=self._device, dtype=torch.float32)
        batch_sims = torch.zeros(len(batch_vecs)).to(self._device)
        batch_xs = torch.zeros(len(batch_vecs)).to(self._device)
        batch_ys = torch.zeros(len(batch_vecs)).to(self._device)

        # CONVOLUTION IDEAS
        # goal is to find cos(theta) = A . B / (||A|| * ||B||)
        # - first do convolution between batch_vecs (tensor) and norm_query_features*mask_tensor (kernel)
        # - batch_vecs is not normalized, so we need to find vector mag. for each window we used
        #   - this can be done by first doing batch_vecs * batch_vecs (element-wise)
        #   - then, we can do a second convolution between squared vecs and the mask tensor to get squared magnitude
        #   - then just divide convolution outputs element-wise

        scaledSims = torch.conv2d(batch_vecs, norm_query_features * mask_tensor)

        sq_batch_vecs = batch_vecs * batch_vecs
        sq_mask_tensor = mask_tensor * mask_tensor
        batch_mags = torch.conv2d(sq_batch_vecs.view(-1, 1, height, width), sq_mask_tensor)
        batch_mags = batch_mags.view(batch_vecs.shape[0], 
                                     batch_vecs.shape[1],
                                     height - q_height + 1,
                                     width - q_width + 1)
        batch_mags = torch.sum(batch_mags, 1, keepdim=True)
        batch_mags = torch.sqrt(batch_mags) + 1e-5 # add small eps to avoid NaN values

        window_sims = scaledSims / batch_mags
        window_sims = window_sims.view(window_sims.shape[0], -1)

        batch_sims, idxs = window_sims.max(dim=1)
        batch_xs = idxs % (width - q_width + 1)
        batch_ys = torch.div(idxs, width - q_width + 1, rounding_mode='floor')

        return batch_sims, batch_xs, batch_ys

class LiveSearchTool(SearchTool):
    '''Implementation of `SearchTool` that computes features on the fly. 
       Does not require a precomputed feature cache, but should only be used with
       small/medium datasets.'''
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
    
    @torch.no_grad()
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

class CachedSearchTool(SearchTool):
    '''Implementation of `SearchTool` that uses a precomputed cache to efficiently 
       compute search results. See `caching.py` for creating a new cache.''' 
    def __init__(self, model, cache: zarr.Array | torch.Tensor | np.ndarray, device, batch_size=500):
        super().__init__(model, device)
        self._cache = cache
        self._batch_size = batch_size

    @torch.no_grad()
    def compute(self, query_mask):
        sims = []
        xs = []
        ys = []
        for i in range(0, len(self._cache), self._batch_size):
            batch_arr = self._cache[i:i + self._batch_size]
            batch_sims, batch_xs, batch_ys = self.compute_batch(query_mask, batch_arr)
            sims.append(batch_sims)
            xs.append(batch_xs)
            ys.append(batch_ys)
        return torch.cat(sims), torch.cat(xs), torch.cat(ys)
