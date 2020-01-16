import math

import torch


def normalize(data, axis):
    return data / torch.norm(data, p=None, dim=axis, keepdim=True)


def compute_viewpoints(azimuth, elevation, distance):
    if isinstance(azimuth, float) or isinstance(azimuth, int):
        x = distance * math.cos(elevation) * math.sin(azimuth)
        y = distance * math.sin(elevation)
        z = -distance * math.cos(elevation) * math.cos(azimuth)
        raise NotImplementedError
    else:
        x = distance * torch.cos(elevation) * torch.sin(azimuth)
        y = distance * torch.sin(elevation)
        z = -distance * torch.cos(elevation) * torch.cos(azimuth)
    if max(x.ndim, y.ndim, z.ndim) == 0:
        # single viewpoint
        viewpoints = torch.stack((x, y, z))
    else:
        # batch
        if x.ndim != y.ndim or y.ndim != z.ndim:
            raise NotImplementedError
        viewpoints = torch.stack((x, y, z), axis=1)

    return viewpoints


def get_dtype_in_cuda(dtype):
    dtype = str(dtype)
    if dtype == 'float32':
        return 'float'
    elif dtype == 'float64':
        return 'double'
    elif dtype == 'int32':
        return 'int'
    elif dtype == 'int64':
        return 'long'
    else:
        raise NotImplementedError


def broadcast(*tensor_list):
    min_ndim = min([t.ndim for t in tensor_list])
    max_ndim = max([t.ndim for t in tensor_list])
    if min_ndim == max_ndim:
        return tensor_list

    batch_sizes = set([t.shape[0] for t in tensor_list if t.ndim == max_ndim])
    assert len(batch_sizes) == 1
    batch_size = batch_sizes.pop()

    shape = [batch_size] + list(map(max, zip(*[t.shape[-min_ndim:] for t in tensor_list])))

    tensor_list_new = []
    for i, t in enumerate(tensor_list):
        if t.ndim == max_ndim:
            tensor_list_new.append(t.expand(*shape))
        else:
            tensor_list_new.append(t[None].expand(*shape))

    return tensor_list_new
