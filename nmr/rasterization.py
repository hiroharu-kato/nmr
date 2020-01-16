import string

import chainer_pytorch_migration as cpm
import cupy as cp
import torch

from . import utils


def distribute(data, indices, foreground_maps, is_batch_data=False, is_batch_indices=False, default_value=-1):
    if not is_batch_data:
        data = data[indices]
    else:
        if not is_batch_indices:
            data = data[:, indices]
        else:
            data = torch.stack([d[i] for d, i in zip(data, indices)])
    data = mask(data, foreground_maps, default_value)
    return data


class Mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, masks, default_value):
        # PyTorch to CuPy
        data_in = cp.asarray(data)
        masks = cp.asarray(masks)
        data_out = data_in.copy()
        dim = data_in.size / masks.size

        # distribute
        masks = cp.ascontiguousarray(masks)
        data_out = cp.ascontiguousarray(data_out)
        kernel = cp.ElementwiseKernel(
            'raw S data_out, int64 mask',
            '',
            string.Template('''
                if (mask == 0) {
                    ${dtype}* p = (${dtype}*)&data_out[i * ${dim}];
                    for (int j = 0; j < ${dim}; j++) *p++ = ${default_value};
                }
            ''').substitute(
                dim=dim,
                dtype=utils.get_dtype_in_cuda(data_out.dtype),
                default_value=default_value,
            ),
            'function',
        )
        kernel(data_out, masks)

        # CuPy to PyTorch
        data_out = cpm.astensor(data_out)

        return data_out

    def backward(self, data, face):
        raise NotImplementedError


class Downsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, foreground_maps):
        # PyTorch to CuPy
        data_in = cp.asarray(data)
        foreground_maps = cp.asarray(foreground_maps)

        # create placeholder of output
        image_h, image_w = data.shape[:2]
        dim = data_in.size // foreground_maps.size
        if dim == 1:
            data_out = cp.zeros((image_h // 2, image_w // 2), dtype=data_in.dtype)
        else:
            data_out = cp.zeros((image_h // 2, image_w // 2, dim), dtype=data_in.dtype)

        # distribute
        data_in = cp.ascontiguousarray(data_in)
        foreground_maps = cp.ascontiguousarray(foreground_maps)
        data_out = cp.ascontiguousarray(data_out)
        loop = cp.arange(image_h / 2 * image_w / 2).astype('int64')
        kernel = cp.ElementwiseKernel(
            'int64 _, raw S data_in, raw int64 foreground_maps',
            'raw S data_out',
            string.Template('''
                int y = i / (${image_w} / 2);
                int x = i % (${image_w} / 2);
                int i1 = (y * 2 + 0) * ${image_w} + (x * 2 + 0);
                int i2 = (y * 2 + 1) * ${image_w} + (x * 2 + 0);
                int i3 = (y * 2 + 0) * ${image_w} + (x * 2 + 1);
                int i4 = (y * 2 + 1) * ${image_w} + (x * 2 + 1);
                int w1 = foreground_maps[i1];
                int w2 = foreground_maps[i2];
                int w3 = foreground_maps[i3];
                int w4 = foreground_maps[i4];
                float w = w1 + w2 + w3 + w4;
                if (w == 0) return;
                for (int j = 0; j < ${dim}; j++) {
                    ${dtype} v = (
                        w1 * data_in[i1 * ${dim} + j] +
                        w2 * data_in[i2 * ${dim} + j] +
                        w3 * data_in[i3 * ${dim} + j] +
                        w4 * data_in[i4 * ${dim} + j]) / w;
                    data_out[i * ${dim} + j] = v;
                }
            ''').substitute(
                image_w=image_w,
                dim=dim,
                dtype=utils.get_dtype_in_cuda(data_in.dtype),
            ),
            'function',
        )
        kernel(loop, data_in, foreground_maps, data_out)

        # CuPy to PyTorch
        data_out = cpm.astensor(data_out)

        return data_out

    def backward(self, data, face):
        raise NotImplementedError


def downsample(data, foreground_maps):
    if foreground_maps is None:
        data = data.type(torch.float32)
        return (data[0::2, 0::2] + data[1::2, 0::2] + data[0::2, 1::2] + data[1::2, 1::2]) / 4
    else:
        is_batch = foreground_maps.ndim == 3
        if is_batch:
            return (data[:, 0::2, 0::2] + data[:, 1::2, 0::2] + data[:, 0::2, 1::2] + data[:, 1::2, 1::2]) / 4
        else:
            return (data[0::2, 0::2] + data[1::2, 0::2] + data[0::2, 1::2] + data[1::2, 1::2]) / 4
    # if foreground_maps is None:
    #     data = data.type(torch.float32)
    #     return (data[0::2, 0::2] + data[1::2, 0::2] + data[0::2, 1::2] + data[1::2, 1::2]) / 4
    # else:
    #     return Downsample.apply(data, foreground_maps)


def mask(data, masks, default_value=0):
    return Mask.apply(data, masks, default_value)


def compute_face_index_maps(vertices, faces, image_h, image_w, near, far, is_batch_vertices):
    # vertices:
    #   - [batch_size, num_vertices, 3] if is_batch_vertices
    #   - [num_vertices, 3]             if not is_batch_vertices
    # faces:
    #   - [num_faces, 3]

    # PyTorch to CuPy
    vertices = cp.asarray(vertices)
    faces = cp.asarray(faces)

    # assertion of shapes
    if is_batch_vertices:
        assert vertices.ndim == 3
    else:
        assert vertices.ndim == 2
    assert faces.ndim == 2
    num_faces = faces.shape[0]

    # face indices to face coordinates
    if is_batch_vertices:
        faces = vertices[:, faces]
    else:
        faces = vertices[faces]

    #
    faces = cp.ascontiguousarray(faces)
    if is_batch_vertices:
        batch_size = vertices.shape[0]
        loop = cp.arange(batch_size * image_h * image_w).astype('int64')
    else:
        loop = cp.arange(image_h * image_w).astype('int64')
    kernel = cp.ElementwiseKernel(
        'int64 _, raw float32 faces',
        'int64 face_index, int64 is_foreground',
        string.Template('''
            const int ih = ${image_h};
            const int iw = ${image_w};
            const int nf = ${num_faces};
            int bn = 0;
            if (${is_batch_vertices}) bn = i / (ih * iw);  // batch number
            const int pn = i % (ih * iw);  // pixel number
            const float yp = ih - (pn / ih + 0.5);
            const float xp = pn % ih + 0.5;
            
            float* face = (float*)&faces[bn * nf * 9];  // pointer to current face
            float depth_min = ${far};
            int face_index_min = -1;
            is_foreground = 0;
            for (int fn = 0; fn < nf; fn++) {
                /* go to next face */
                const float x0 = *face++;
                const float y0 = *face++;
                const float z0 = *face++;
                const float x1 = *face++;
                const float y1 = *face++;
                const float z1 = *face++;
                const float x2 = *face++;
                const float y2 = *face++;
                const float z2 = *face++;
                
                /* continue if (xp, yp) is outside of the rectangle */
                if (xp < x0 && xp < x1 && xp < x2) continue;
                if (x0 < xp && x1 < xp && x2 < xp) continue;
                if (yp < y0 && yp < y1 && yp < y2) continue;
                if (y0 < yp && y1 < yp && y2 < yp) continue;
                
                /* check in or out. w0, w1, w2 should have the same sign. */
                float w0 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1); 
                float w1 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2);
                float w2 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0);
                if (w0 * w1 <= 0) continue;
                if (w1 * w2 <= 0) continue;

                /* normalize w */
                const float w_sum = w0 + w1 + w2;
                w0 /= w_sum;
                w1 /= w_sum;
                w2 /= w_sum;
                
                /* compute 1 / zp = sum(w / z) */
                const float zp = 1. / (w0 / z0 + w1 / z1 + w2 / z2);
                if (zp <= ${near} || ${far} <= zp) continue;
                
                /* check z-buffer */
                if (zp <= depth_min - ${depth_min_delta}) {
                    depth_min = zp;
                    face_index_min = fn;
                    is_foreground = 1;
                }
            }
            /* set to global memory */
            face_index = face_index_min;
        ''').substitute(
            num_faces=num_faces,
            image_h=image_h,
            image_w=image_w,
            near=near,
            far=far,
            is_batch_vertices=int(is_batch_vertices),
            depth_min_delta=1e-4,
        ),
        'function',
    )
    face_index_maps, foreground_maps = kernel(loop, faces)
    if is_batch_vertices:
        face_index_maps = face_index_maps.reshape((-1, image_h, image_w))
        foreground_maps = foreground_maps.reshape((-1, image_h, image_w))
    else:
        face_index_maps = face_index_maps.reshape((image_h, image_w))
        foreground_maps = foreground_maps.reshape((image_h, image_w))

    # CuPy to PyTorch
    face_index_maps = cpm.astensor(face_index_maps)
    foreground_maps = cpm.astensor(foreground_maps)

    return face_index_maps, foreground_maps


def compute_weight_map(vertex_maps, foreground_maps, is_batch):
    if is_batch:
        x0 = vertex_maps[:, :, :, 0, 0]
        x1 = vertex_maps[:, :, :, 1, 0]
        x2 = vertex_maps[:, :, :, 2, 0]
        y0 = vertex_maps[:, :, :, 0, 1]
        y1 = vertex_maps[:, :, :, 1, 1]
        y2 = vertex_maps[:, :, :, 2, 1]
    else:
        x0 = vertex_maps[:, :, 0, 0]
        x1 = vertex_maps[:, :, 1, 0]
        x2 = vertex_maps[:, :, 2, 0]
        y0 = vertex_maps[:, :, 0, 1]
        y1 = vertex_maps[:, :, 1, 1]
        y2 = vertex_maps[:, :, 2, 1]
    image_h, image_w = vertex_maps.shape[-4:-2]
    yp = image_h - (torch.arange(image_h, dtype=torch.float32, device=vertex_maps.device) + 0.5)
    xp = torch.arange(image_w, dtype=torch.float32, device=vertex_maps.device) + 0.5
    yp, xp = torch.broadcast_tensors(yp[:, None], xp[None, :])
    if is_batch:
        yp = yp.unsqueeze(0)
        xp = xp.unsqueeze(0)
    w0 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1)
    w1 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2)
    w2 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0)
    w = torch.stack((w0, w1, w2), dim=-1)
    w = w / w.sum(-1, keepdim=True)
    w = mask(w, foreground_maps)
    return w


def compute_depth_maps(vertex_maps, weight_maps, foreground_maps):
    is_batch_vertices = vertex_maps.ndim == 5
    if is_batch_vertices:
        z_maps = vertex_maps[:, :, :, :, 2]
        z_maps = 1. / (weight_maps / z_maps).sum(3)
        z_maps = mask(z_maps, foreground_maps)
    else:
        z_maps = vertex_maps[:, :, :, 2]
        z_maps = 1. / (weight_maps / z_maps).sum(2)
        z_maps = mask(z_maps, foreground_maps)
    return z_maps


def compute_normal_maps(vertex_n_w_maps, vertex_n_c_maps, vertex_maps, weight_maps, foreground_maps, is_batch):
    normal_w_maps = interpolate(vertex_n_w_maps, vertex_maps, weight_maps, is_batch)
    normal_c_maps = interpolate(vertex_n_c_maps, vertex_maps, weight_maps, is_batch)
    return compute_normal_maps_no_weight(normal_w_maps, normal_c_maps, foreground_maps)


def compute_normal_maps_no_weight(normal_w_maps, normal_c_maps, foreground_maps):
    normal_sign = (normal_c_maps[:, :, 2] <= 0)
    normal_w_maps = normal_w_maps * normal_sign[:, :, None] - normal_w_maps * torch.logical_not(normal_sign)[:, :, None]
    normal_c_maps = normal_c_maps * normal_sign[:, :, None] - normal_c_maps * torch.logical_not(normal_sign)[:, :, None]
    normal_w_maps = normal_w_maps / torch.sqrt(torch.sum(normal_w_maps ** 2, dim=2, keepdim=True))
    normal_c_maps = normal_c_maps / torch.sqrt(torch.sum(normal_c_maps ** 2, dim=2, keepdim=True))
    normal_w_maps = mask(normal_w_maps, foreground_maps)
    normal_c_maps = mask(normal_c_maps, foreground_maps)
    return normal_w_maps, normal_c_maps


def compute_color_maps(vertex_t_maps, textures, texture_params_maps, foreground_maps, is_batch_vertices):
    is_batch_textures = textures.ndim == 4
    texture_height, texture_width = textures.shape[-3:-1]
    y_max = texture_params_maps.select(-1, 0)
    x_max = texture_params_maps.select(-1, 1)
    y_offset = texture_params_maps.select(-1, 2)
    ty_f = (1 - vertex_t_maps.select(-1, 1)) * y_max + y_offset
    tx_f = vertex_t_maps.select(-1, 0) * x_max
    ty_i_f = torch.floor(ty_f).type(torch.int64).clamp(0, texture_height - 1)
    ty_i_c = torch.ceil(ty_f).type(torch.int64).clamp(0, texture_height - 1)
    tx_i_f = torch.floor(tx_f).type(torch.int64).clamp(0, texture_width - 1)
    tx_i_c = torch.ceil(tx_f).type(torch.int64).clamp(0, texture_width - 1)
    t_i_ff = ty_i_f * texture_width + tx_i_f
    t_i_fc = ty_i_f * texture_width + tx_i_c
    t_i_cf = ty_i_c * texture_width + tx_i_f
    t_i_cc = ty_i_c * texture_width + tx_i_c
    w_ff = (1 - (ty_f - ty_i_f)) * (1 - (tx_f - tx_i_f))
    w_fc = (1 - (ty_f - ty_i_f)) * (1 - (tx_i_c - tx_f))
    w_cf = (1 - (ty_i_c - ty_f)) * (1 - (tx_f - tx_i_f))
    w_cc = (1 - (ty_i_c - ty_f)) * (1 - (tx_i_c - tx_f))
    w_sum = w_ff + w_fc + w_cf + w_cc
    w_ff = w_ff / w_sum
    w_fc = w_fc / w_sum
    w_cf = w_cf / w_sum
    w_cc = w_cc / w_sum
    if is_batch_textures:
        t2 = textures.reshape((textures.shape[0], -1, 3))
    else:
        t2 = textures.reshape((-1, 3))
    t_ff = distribute(t2, t_i_ff, foreground_maps, is_batch_textures, is_batch_vertices, default_value=0.)
    t_fc = distribute(t2, t_i_fc, foreground_maps, is_batch_textures, is_batch_vertices, default_value=0.)
    t_cf = distribute(t2, t_i_cf, foreground_maps, is_batch_textures, is_batch_vertices, default_value=0.)
    t_cc = distribute(t2, t_i_cc, foreground_maps, is_batch_textures, is_batch_vertices, default_value=0.)
    color_maps = (
            t_ff * w_ff.unsqueeze(-1) + t_fc * w_fc.unsqueeze(-1) +
            t_cf * w_cf.unsqueeze(-1) + t_cc * w_cc.unsqueeze(-1))
    color_maps = mask(color_maps, foreground_maps)
    return color_maps


def reflectance_maps(normal_w_maps, normal_c_maps):
    return torch.relu(normal_w_maps.select(-1, 1)) * 0.3 + 0.7
    # return torch.relu(-normal_c_maps[:, :, 2]) * 0.5 + 0.5


def compute_normals(vertices, faces):
    vs = vertices[faces]
    v0 = vs[:, 0]
    v1 = vs[:, 1]
    v2 = vs[:, 2]
    e01 = v1 - v0
    e02 = v2 - v0
    normals = torch.cross(e01, e02)
    normals * torch.rsqrt((normals ** 2).sum(axis=1, keepdims=True))
    return normals


def interpolate(data, vertex_maps, weight_maps, is_batch):
    if is_batch:
        a = ((data / vertex_maps[:, :, :, :, 2:]) * weight_maps[:, :, :, :, None]).sum(3)
        b = ((1 / vertex_maps[:, :, :, :, 2:]) * weight_maps[:, :, :, :, None]).sum(3)
    else:
        a = ((data / vertex_maps[:, :, :, 2:]) * weight_maps[:, :, :, None]).sum(2)
        b = ((1 / vertex_maps[:, :, :, 2:]) * weight_maps[:, :, :, None]).sum(2)
    return a / b
