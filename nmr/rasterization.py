import string
import cupy as cp
import chainer_pytorch_migration as cpm
import torch
from . import utils


class Distribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, indices, is_batch_data=False, is_batch_indices=False, default_value=-1):
        # data:
        #   - [batch_size, num_data, *data] if is_batch_data
        #   - [num_data, *data]             if not is_batch_data
        # indices:
        #   - [batch_size, *indices]        if is_batch_indices
        #   - [*indices]                    if not is_batch_indices

        # PyTorch to CuPy
        data_in = cp.asarray(data)
        indices = cp.asarray(indices)

        # assert shapes
        if is_batch_data:
            assert 2 <= data.ndim
            batch_size = data_in.shape[0]
            num_data = data_in.shape[1]
            shape_data = data_in.shape[2:]
            dim_data = data_in.size / (batch_size * num_data)
        else:
            assert 1 <= data.ndim
            num_data = data_in.shape[0]
            shape_data = data_in.shape[1:]
            dim_data = data_in.size / num_data
        if is_batch_indices:
            dim_indices = indices.size / indices.shape[0]
        else:
            dim_indices = indices.size
        assert indices.max() < num_data

        # create placeholder of output
        if dim_data == 1:
            data_out = cp.ones(indices.shape, dtype=data_in.dtype)
        else:
            data_out = cp.ones(tuple(list(indices.shape) + list(shape_data)), dtype=data_in.dtype)
        data_out = data_out * default_value

        # distribute
        data_in = cp.ascontiguousarray(data_in)
        indices = cp.ascontiguousarray(indices)
        data_out = cp.ascontiguousarray(data_out)
        kernel = cp.ElementwiseKernel(
            'raw S data_in, int64 index, raw S data_out',
            '',
            string.Template('''
                if (index < 0) return;
                int pos_from = index * ${dim_data};
                if (${is_batch_data}) {
                    int bn = i / ${dim_indices};
                    pos_from += bn * ${num_data} * ${dim_data};
                }
                int pos_to = i * ${dim_data};
                ${dtype}* p1 = (${dtype}*)&data_in[pos_from];
                ${dtype}* p2 = (${dtype}*)&data_out[pos_to];
                for (int j = 0; j < ${dim_data}; j++) *p2++ = *p1++;
            ''').substitute(
                is_batch_data=int(is_batch_data),
                is_batch_indices=int(is_batch_indices),
                num_data=num_data,
                dim_data=dim_data,
                dim_indices=dim_indices,
                dtype=utils.get_dtype_in_cuda(data_in.dtype),
            ),
            'function',
        )
        kernel(data_in, indices, data_out)

        # CuPy to PyTorch
        data_out = cpm.astensor(data_out)

        return data_out

    def backward(self, data, face):
        pass


def distribute(data, indices, is_batch_data=False, is_batch_indices=False, default_value=-1):
    return Distribute.apply(data, indices, is_batch_data, is_batch_indices, default_value)


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
        'int64 face_index, bool is_foreground',
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
                if (zp <= depth_min) {
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


def compute_weight_map(vertex_maps, foreground_maps):
    x0 = vertex_maps[:, :, 0, 0]
    x1 = vertex_maps[:, :, 1, 0]
    x2 = vertex_maps[:, :, 2, 0]
    y0 = vertex_maps[:, :, 0, 1]
    y1 = vertex_maps[:, :, 1, 1]
    y2 = vertex_maps[:, :, 2, 1]
    if vertex_maps.ndim == 4:
        image_h, image_w = vertex_maps.shape[:2]
        yp = image_h - (torch.arange(image_h, dtype=torch.float32, device=vertex_maps.device) + 0.5)
        xp = torch.arange(image_w, dtype=torch.float32, device=vertex_maps.device) + 0.5
        yp, xp = torch.broadcast_tensors(yp[:, None], xp[None, :])
    w0 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1)
    w1 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2)
    w2 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0)
    w = torch.stack((w0, w1, w2), dim=2)
    w = w / w.sum(-1, keepdim=True)
    # w = mask(w, foreground_maps)
    return w