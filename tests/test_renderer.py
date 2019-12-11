

import unittest
import nmr
import torch


class TestRenderer(unittest.TestCase):
    def test(self):
        vertices, _, vertices_n, faces, _, faces_n, _ = nmr.load_obj('./data/models/teapot.obj')
        vertices = torch.as_tensor(vertices).cuda()
        vertices_n = torch.as_tensor(vertices_n).cuda()
        faces = torch.as_tensor(faces).cuda()
        faces_n = torch.as_tensor(faces_n).cuda()
        meshes = nmr.create_meshes(vertices=vertices, vertices_n=vertices_n, faces=faces, faces_n=faces_n)
        import IPython
        IPython.embed()
        # nmr.load_obj('./data/models/1a6fca5e59b00eba250a73fdbcda6406/model.obj')


if __name__ == '__main__':
    unittest.main()