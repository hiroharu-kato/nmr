from . import utils


class Meshes(object):
    def __init__(
            self, vertices=None, vertices_t=None, normals=None, faces=None, faces_t=None, faces_n=None,
            textures=None, texture_params=None):
        # Vertices (vertices) must be [batch_size, num_vertices, 3].
        # Texture vertices (vertices_t) must be [batch_size, num_vertices_t, 2].
        # Normal vectors (normals) must be [batch_size, num_normals, 3].
        # Normal vectors can be undefined.
        utils.assert_shape(vertices, (None, None, 3))
        utils.assert_shape(vertices_t, (None, None, 2))
        if normals is not None:
            utils.assert_shape(normals, (None, None, 3))

        # Indices assigned to faces (faces, faces_t, faces_n) must be [num_faces, 3].
        # faces_n can be undefined.
        utils.assert_shape(faces, (None, 3))
        utils.assert_shape(faces_t, (None, 3))
        if faces_n is not None:
            utils.assert_shape(faces_n, (None, 3))

        # Texture images (textures) must be [batch_size, height, width, 3].
        utils.assert_shape(textures, (None, None, None, 3))

        # TODO: assertion for texture_params.

        self.vertices = vertices
        self.vertices_t = vertices_t
        self.normals = normals
        self.faces = faces
        self.faces_t = faces_t
        self.faces_n = faces_n
        self.textures = textures
        self.texture_params = texture_params


def create_meshes(
        vertices=None, vertices_t=None, normals=None, faces=None, faces_t=None, faces_n=None, textures=None,
        texture_params=None):
    return Meshes(vertices, vertices_t, normals, faces, faces_t, faces_n, textures, texture_params)
