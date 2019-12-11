import string
import cupy as cp


def compute_face_index_maps(vertices, faces, image_h, image_w, near=0.1, far=100, eps=1e-8):
    assert vertices.ndim == 3
    assert faces.ndim == 2

    vertices = cp.asarray(vertices)
    faces = cp.asarray(faces)

    batch_size, num_vertices = vertices.shape[:2]
    num_faces = faces.shape[0]

    faces = vertices[:, faces]

    faces = cp.ascontiguousarray(faces)
    loop = cp.arange(batch_size * image_h * image_w).astype('int64')
    kernel = cp.ElementwiseKernel(
        'int64 _, raw float32 faces',
        'int64 face_index',
        string.Template('''
            const float eps = ${eps};
            const int ih = ${image_h};
            const int iw = ${image_w};
            const int nf = ${num_faces};
            const int bn = i / (ih * iw);  // batch number
            const int pn = i % (ih * iw);  // pixel number
            const int yi = pn / ih;
            const int xi = pn % ih;
            const float yp = yi + 0.5;
            const float xp = xi + 0.5;
            
            float* face = (float*)&faces[bn * nf * 9];  // pointer to current face
            float depth_min = ${far};
            int face_index_min = -1;
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
                if (xp < x0 && xp < x1 && xp < x2) continue;
                if (x0 < xp && x1 < xp && x2 < xp) continue;
                if (yp < y0 && yp < y1 && yp < y2) continue;
                if (y0 < yp && y1 < yp && y2 < yp) continue;
                
                /* compute w */
                float det = x2 * (y0 - y1) + x0 * (y1 - y2) + x1 * (y2 - y0);
                float w[3];
                w[0] = (yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1)) / det;
                w[1] = (yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2)) / det;
                w[2] = (yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0)) / det;
                if (w[0] < 0 || 1 < w[0] || w[1] < 0 || 1 < w[1] || w[2] < 0 || 1 < w[2]) continue;
                const float w_sum = w[0] + w[1] + w[2];
                w[0] /= w_sum;
                w[1] /= w_sum;
                w[2] /= w_sum;
                
                /* compute 1 / zp = sum(w / z) */
                const float zp = 1. / (w[0] / z0 + w[1] / z1 + w[2] / z2);
                if (zp <= ${near} || ${far} <= zp) continue;
                
                /* check z-buffer */
                if (zp <= depth_min) {
                    depth_min = zp;
                    face_index_min = fn;
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
            eps=eps,
        ),
        'function',
    )
    face_index_map = kernel(loop, faces)
    face_index_map = face_index_map.reshape((batch_size, image_h, image_w))
    return face_index_map
    
    # import skimage.io
    # a = face_index_map[0]
    # b = (a >= 0).astype('float32').get()
    # print(b.mean())
    # print(a[0, 0])
    # skimage.io.imsave('a.png', b)
