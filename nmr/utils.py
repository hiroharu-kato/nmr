import torch


def assert_shape(data, shape):
    assert isinstance(data, torch.Tensor)
    assert data.ndim == len(shape)
    for s1, s2 in zip(shape, data.shape):
        if s1 is not None:
            assert s1 == s2


def normalize(data, axis):
    return data / torch.norm(data, p=None, dim=axis, keepdim=True)


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
