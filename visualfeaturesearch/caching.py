import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

# function for getting the 4D image tensors from each batch
_default_data_img_map = lambda batch : batch[0]
 
def precompute(dataloader: DataLoader, model, cache_path, array_name, device, dtype=np.float32, data_img_map=_default_data_img_map):
    # get output dimensions of the intermediate layer
    img0 = data_img_map(dataloader.dataset[0])
    img0 = img0.to(device)
    output0 = model(img0[None, :, :, :])
    tmpfs = output0.shape
    feature_shape = (len(dataloader.dataset), tmpfs[1], tmpfs[2], tmpfs[3])

    # create caching store with Zarr
    store = zarr.DirectoryStore(cache_path)
    root = zarr.group(store=store, overwrite=False)
    out_feats = root.zeros(array_name,
            shape=feature_shape,
            chunks=(500, None, None, None),
            dtype=dtype,
            overwrite=True)

    with torch.no_grad():
        it = iter(dataloader)
        idx = 0
        for batch in it:
            # for each batch, compute features & save to Zarr store
            batch = data_img_map(batch).to(device)
            features = model(batch)
            features = features.cpu().numpy().astype(dtype)
            length = min(dataloader.batch_size, len(out_feats) - idx)
            out_feats[idx:idx+length] = features[:length]
            idx += length
            print('Progress:', idx, '/', len(dataloader.dataset))
            del batch, features