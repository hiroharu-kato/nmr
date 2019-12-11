from . import rasterization

class Renderer(object):
    def __init__(self):
        pass

    def __call__(self, meshes, cameras, lights):
        vertices = cameras(meshes.vertices)

        vertices = vertices[None, :, :]
        rasterization.compute_face_index_maps(vertices, meshes.faces, cameras.image_h, cameras.image_w)