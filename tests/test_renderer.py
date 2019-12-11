

import unittest
import nmr
import torch


class TestRenderer(unittest.TestCase):
    def test(self):
        filename = './data/obj/1a6fca5e59b00eba250a73fdbcda6406/model.obj'
        image_h = image_w = 224

        vertices, _, vertices_n, faces, _, faces_n, _ = nmr.load_obj(filename)
        vertices = torch.as_tensor(vertices).cuda()
        vertices_n = torch.as_tensor(vertices_n).cuda()
        faces = torch.as_tensor(faces).cuda()
        faces_n = torch.as_tensor(faces_n).cuda()
        meshes = nmr.create_meshes(vertices=vertices, vertices_n=vertices_n, faces=faces, faces_n=faces_n)

        camera_origin = torch.as_tensor([0, 0, -2], dtype=torch.float32).cuda()
        cameras = nmr.create_cameras(origin=camera_origin, image_h=image_h, image_w=image_w)

        renderer = nmr.Renderer()
        renderer(meshes, cameras, None)

if __name__ == '__main__':
    unittest.main()