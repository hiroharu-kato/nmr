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
