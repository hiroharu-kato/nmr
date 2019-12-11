import os
import warnings
import numpy as np
import collections
import skimage.io


def load_textures(filename):
    materials = collections.OrderedDict()
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        k = line.split()[0]
        vs = line.split()[1:]
        if k == 'newmtl':
            material_name = line.split()[1]
            materials[material_name] = {}
        elif k == 'Kd':
            materials[material_name]['Kd'] = np.array(list(map(float, vs)), np.float32)
        elif k == 'map_Kd':
            materials[material_name]['map_Kd'] = os.path.join(os.path.dirname(filename), vs[0])

    textures = list()
    for name, material in materials.items():
        if 'map_Kd' in material:
            texture = skimage.io.imread(material['map_Kd']).astype(np.float32) / 255.
            texture = texture[:, :, :3]
        elif 'Kd' in material:
            texture = np.ones((1, 1, 3), np.float32) * material['Kd']
        else:
            # default color is white
            texture = np.ones((1, 1, 3), np.float32)
        textures.append(texture)

    max_width = max([t.shape[1] for t in textures])
    textures = [np.pad(t, ((0, 0), (0, max_width - t.shape[1]), (0, 0))) for t in textures]
    y_offsets = np.cumsum([0] + [t.shape[0] for t in textures])[:-1]
    image_sizes = [t.shape[:2] for t in textures]
    y_offsets = {n: o for n, o in zip(materials.keys(), y_offsets)}
    image_sizes = {n: s for n, s in zip(materials.keys(), image_sizes)}
    textures = np.concatenate(textures, axis=0)

    return textures, image_sizes, y_offsets


def load_obj(filename):
    faces = []
    faces_t = []
    faces_n = []
    vertices = []
    vertices_n = []
    vertices_t = []
    textures = None
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue

        k = line.split()[0]
        vs = line.split()[1:]
        if k == 'mtllib':
            filename_mtl = '%s/%s' % (os.path.dirname(filename), vs[0])
            textures, image_sizes, y_offsets = load_textures(filename_mtl)
        elif k == 'usemtl':
            material_name = vs[0]
        elif k == 'g':
            # group
            pass
        elif k == 'v':
            vertices.append(list(map(float, vs)))
        elif k == 'vn':
            vertices_n.append(list(map(float, vs)))
        elif k == 'vt':
            y_offset = y_offsets[material_name]
            image_size = image_sizes[material_name]
            uv = np.array(list(map(float, vs)))
            uv[0] = (uv[0] % 1) * (image_size[0] - 1) + y_offset
            uv[1] = (uv[1] % 1) * (image_size[1] - 1)
            vertices_t.append(uv)
        elif k == 'f':
            # v1/vt1/vn1
            f = list(map(int, [v.split('/')[0] for v in vs]))
            faces.append(f)
            if len(vs[0].split('/')[1]):
                ft = list(map(int, [v.split('/')[1] for v in vs]))
                faces_t.append(ft)
            if len(vs[0].split('/')[2]):
                fn = list(map(int, [v.split('/')[2] for v in vs]))
                faces_n.append(fn)
        else:
            warnings.warn('unsupported option: %s.' % k)

    vertices = np.array(vertices, np.float32)
    vertices_t = np.array(vertices_t, np.float32)
    if not vertices_t.size:
        vertices_t = None
    vertices_n = np.array(vertices_n, np.float32)
    if not vertices_n.size:
        vertices_n = None
    faces = np.array(faces, np.int32) - 1
    faces_t = np.array(faces_t, np.int32) - 1
    if not faces_t.size:
        faces_t = None
    faces_n = np.array(faces_n, np.int32) - 1
    if not faces_n.size:
        faces_n = None
    return vertices, vertices_t, vertices_n, faces, faces_t, faces_n, textures