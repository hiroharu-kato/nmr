from . import utils


class Meshes(object):
    def __init__(
            self, vertices=None, vertices_t=None, normals=None, faces=None, faces_t=None, faces_n=None,
            textures=None, texture_params=None):
        # Vertices (vertices) must be [num_vertices, 3].
        # Texture vertices (vertices_t) must be [num_vertices_t, 2].
        # Normal vectors (normals) must be [num_normals, 3].
        # These can be minibatch.
        # Normal vectors can be undefined.
        vertices = utils.assert_shape(vertices, (None, 3), True)
        vertices_t = utils.assert_shape(vertices_t, (None, 2), True)
        if normals is not None:
            normals = utils.assert_shape(normals, (None, 3), True)

        # Indices assigned to faces (faces, faces_t, faces_n) must be [num_faces, 3].
        # faces_n can be undefined.
        # These cannot be minibatch.
        faces = utils.assert_shape(faces, (None, 3), False)
        faces_t = utils.assert_shape(faces_t, (None, 3), False)
        if faces_n is not None:
            faces_n = utils.assert_shape(faces_n, (None, 3), False)

        # Texture images (textures) must be [height, width, 3] and can be minibatch.
        textures = utils.assert_shape(textures, (None, None, 3), True)

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
