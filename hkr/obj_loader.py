import os
import warnings
import numpy as np


def load_mtl(filename):
    materials = {}
    material_name = ''
    for line in open(filename).readlines():
        line = line.strip()
        if not len(line):
            continue
        elif line.startswith('#'):
            continue
        k = line.split()[0]
        vs = line.split()[1:]
        if k == 'newmtl':
            material_name = line.split()[1]
            materials[material_name] = {}
        elif k == 'Ka':
            materials[material_name]['Ka'] = np.array(list(map(float, vs)), np.float32)
        elif k == 'Kd':
            materials[material_name]['Kd'] = np.array(list(map(float, vs)), np.float32)
        elif k == 'Ks':
            materials[material_name]['Ks'] = np.array(list(map(float, vs)), np.float32)
        elif k == 'd':
            materials[material_name]['dissolve'] = vs[0]
        elif k == 'map_Kd':
            materials[material_name]['map_Kd'] = vs[0]
        else:
            warnings.warn('Not supported: %s.' % k)
    return materials


def load_obj(filename):
    vertices = []
    vertices_n = []
    vertices_t = []
    for line in open(filename).readlines():
        line = line.strip()
        if not len(line):
            continue
        elif line.startswith('#'):
            continue
        k = line.split()[0]
        vs = line.split()[1:]
        if k == 'mtllib':
            filename_mtl = '%s/%s' % (os.path.dirname(filename), vs[0])
            materials = load_mtl(filename_mtl)
        elif k == 'usemtl':
            material = materials[vs[0]]
        elif k == 'g':
            # group
            pass
        elif k == 'v':
            vertices.append(list(map(float, vs)))
        elif k == 'vn':
            vertices_n.append(list(map(float, vs)))
        elif k == 'vt':
            vertices_t.append(list(map(float, vs)))
        else:
            warnings.warn('Not supported: %s.' % k)
    import IPython
    IPython.embed()

load_obj('./data/models/1a6fca5e59b00eba250a73fdbcda6406/model.obj')
