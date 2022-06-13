import torch
from torch.utils.data import DataLoader
import zarr
 
def precompute(dataloader: DataLoader, model, cache_path, array_name):
    if not torch.cuda.is_available():
        raise Exception('No GPU Available')

    gpu = torch.device('cuda:0')

    # get output dimensions of the intermediate layer
    img0 = dataloader.dataset[0][0]
    img0 = img0.to(gpu)
    output0 = model(img0[None, :, :, :])
    tmpfs = output0.shape
    feature_shape = (len(dataloader.dataset), tmpfs[1], tmpfs[2], tmpfs[3])

    # create caching store with Zarr
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
            # for each batch, compute features & save to Zarr store
            batch = batch[0].to(gpu)
            features = model(batch)
            features = features.cpu().numpy()
            length = min(dataloader.batch_size, len(out_feats) - idx)
            out_feats[idx:idx+length] = features[:length]
            idx += length
            print('Progress:', idx, '/', len(dataloader.dataset))
            del batch, features