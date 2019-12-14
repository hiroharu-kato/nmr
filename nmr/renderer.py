from . import rasterization


class Renderer(object):
    def __init__(self):
        self.near = 0.1
        self.far = 1000
        self.anti_aliasing = True

    def __call__(self, meshes, cameras, lights):
        vertices = meshes.vertices
        faces = meshes.faces
        is_batch_vertices = vertices.ndim == 3

        # world coordinates to screen coordinates
        vertices = cameras(vertices)
        if self.anti_aliasing:
            vertices = vertices * 2
            image_h = cameras.image_h * 2
            image_w = cameras.image_w * 2
        else:
            image_h = cameras.image_h
            image_w = cameras.image_w

        # face_index_maps: [batch_size, image_h, image_w]
        # foreground_maps: [batch_size, image_h, image_w]
        # non-differentiable
        face_index_maps, foreground_maps = rasterization.compute_face_index_maps(
            vertices, faces, image_h, image_w, self.near, self.far, is_batch_vertices)

        # vertex_index_maps: [batch_size, image_h, image_w, 3]
        # differentiable w.r.t faces
        vertex_index_maps = rasterization.distribute(faces, face_index_maps, False, is_batch_vertices, -1)

        # vertex_maps: [batch_size, image_h, image_w, (p0, p1, p2), (x, y, z)]
        # differentiable w.r.t vertices
        vertex_maps = rasterization.distribute(
            vertices, vertex_index_maps, is_batch_vertices, is_batch_vertices, default_value=0.)

        # weight_maps: [batch_size, image_h, image_w, (p0, p1, p2)]
        # differentiable w.r.t vertex_maps
        weight_maps = rasterization.compute_weight_map(vertex_maps, foreground_maps)

        

        import skimage.io
        a = face_index_maps
        b = foreground_maps.cpu().numpy().astype('float32')
        print(b.mean())
        if self.anti_aliasing:
            b = (b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2]) / 4
        b = (b * 255).astype('uint8')
        skimage.io.imsave('a.png', b)
