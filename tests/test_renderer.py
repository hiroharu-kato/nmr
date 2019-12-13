import unittest
import numpy as np
import nmr
import torch
import chainer_pytorch_migration as cpm

cpm.use_torch_in_cupy_malloc()


class TestRenderer(unittest.TestCase):
    def test(self):
        filename_obj = './data/obj/1a6fca5e59b00eba250a73fdbcda6406/model.obj'
        filename_views = '/home/hkato/Dropbox/lab/code/nmr/data/obj/1a6fca5e59b00eba250a73fdbcda6406/view.txt'
        view_num = 3
        image_h = image_w = 224

        vertices, _, vertices_n, faces, _, faces_n, _ = nmr.load_obj(filename_obj)
        vertices[:, 2] *= -1
        vertices = torch.as_tensor(vertices).cuda()
        vertices_n = torch.as_tensor(vertices_n).cuda()
        faces = torch.as_tensor(faces).cuda()
        faces_n = torch.as_tensor(faces_n).cuda()
        meshes = nmr.create_meshes(vertices=vertices, vertices_n=vertices_n, faces=faces, faces_n=faces_n)

        viewpoints = open(filename_views).readlines()[view_num]
        azimuth, elevation, _, distance = map(float, viewpoints.split())
        azimuth = np.radians(azimuth + 90)
        elevation = np.radians(elevation)
        azimuth = torch.as_tensor(azimuth, dtype=torch.float32).cuda()
        elevation = torch.as_tensor(elevation, dtype=torch.float32).cuda()
        distance = torch.as_tensor(distance, dtype=torch.float32).cuda()
        viewpoints = nmr.compute_viewpoints(azimuth, elevation, distance)
        cameras = nmr.create_cameras(origin=viewpoints, image_h=image_h, image_w=image_w)

        renderer = nmr.Renderer()
        renderer(meshes, cameras, None)

if __name__ == '__main__':
    unittest.main()