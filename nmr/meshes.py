class Meshes(object):
    def __init__(
            self, vertices=None, vertices_t=None, normals=None, faces=None, faces_t=None, faces_n=None,
            textures=None, texture_params=None):
        # Vertices (vertices) must be [num_vertices, 3] or [batch_size, num_vertices, 3].
        assert vertices.ndim in (2, 3)
        assert vertices.shape[-1] == 3

        # Texture vertices (vertices_t) must be [num_vertices_t, 2] or [batch_size, num_vertices_t, 2].
        assert vertices_t.ndim in (2, 3)
        assert vertices_t.shape[-1] == 2

        # Normal vectors (normals) must be [num_normals, 3] or [batch_size, num_normals, 3].
        # This can be undefined.
        if normals is not None:
            assert normals.ndim in (2, 3)
            assert normals.shape[-1] == 3

        # Vertex indices of faces (faces) must be [num_faces, 3].
        assert faces.ndim == 2
        assert faces.shape[-1] == 3

        # Texture vertex indices of faces (faces_t) must be [num_faces, 3].
        assert faces_t.ndim == 2
        assert faces_t.shape[-1] == 3

        # Texture vertex indices of faces (faces_t) must be [num_faces, 3].
        # This can be undefined.
        if faces_n is not None:
            assert faces_n.ndim == 2
            assert faces_n.shape[-1] == 3

        # Texture images (textures) must be [height, width, 3] or [batch_size, height, width, 3].
        # TODO: swap axies to [batch_size, 3, height, width]
        assert textures.ndim in (3, 4)
        assert textures.shape[-1] == 3

        # TODO: assertion for texture_params.

        self.vertices = vertices
        self.vertices_t = vertices_t
        self.normals = normals
        self.faces = faces
        self.faces_t = faces_t
        self.faces_n = faces_n
        self.textures = textures
        self.texture_params = texture_params


def create_meshes(vertices=None, vertices_t=None, normals=None, faces=None, faces_t=None, faces_n=None,
                  textures=None, texture_params=None):
    return Meshes(vertices, vertices_t, normals, faces, faces_t, faces_n, textures, texture_params)
