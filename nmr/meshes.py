class Meshes(object):
    def __init__(
            self, vertices=None, vertices_t=None, vertices_n=None, faces=None, faces_t=None, faces_n=None,
            textures=None):
        self.vertices = vertices
        self.vertices_t = vertices_t
        self.vertices_n = vertices_n
        self.faces = faces
        self.faces_t = faces_t
        self.faces_n = faces_n
        self.textures = textures


def create_meshes(vertices=None, vertices_t=None, vertices_n=None, faces=None, faces_t=None, faces_n=None,
                  textures=None):
    return Meshes(vertices, vertices_t, vertices_n, faces, faces_t, faces_n, textures)