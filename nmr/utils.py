import numpy as np
import tinyobjloader
import torch

def load_obj(filename):
    r = tinyobjloader.ObjReader()
    assert r.ParseFromFile(filename)

    # load attributes of vertices
    attributes = r.GetAttrib()
    colors = np.array(attributes.colors, np.float32).reshape((-1, 3))
    normals = np.array(attributes.normals, np.float32).reshape((-1, 3))
    texcoords = np.array(attributes.texcoords, np.float32).reshape((-1, 2))
    vertices = np.array(attributes.vertices, np.float32).reshape((-1, 3))

    # load attributes of faces
    shapes = r.GetShapes()
    vertex_indices = np.array(
        [i.vertex_index for s in shapes for i in s.mesh.indices], np.int32).reshape((-1, 3))
    texcoord_indices = np.array(
        [i.texcoord_index for s in shapes for i in s.mesh.indices], np.int32).reshape((-1, 3))
    normal_indices = np.array(
        [i.normal_index for s in shapes for i in s.mesh.indices], np.int32).reshape((-1, 3))


def normalize(data, axis):
    return data / torch.norm(data, p=None, dim=axis, keepdim=True)
