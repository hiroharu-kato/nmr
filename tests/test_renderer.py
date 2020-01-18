import math
import os
import unittest

# Dependency on cpm can be removed when new PyTorch with
# [this pull request](https://github.com/pytorch/pytorch/pull/24947) is released.
# This is described in the "CuPy bridge" section of the Chainer-PyTorch Migration Guide.
import chainer_pytorch_migration as cpm
import numpy as np
import skimage
import torch
import tqdm

import nmr

cpm.use_torch_in_cupy_malloc()


class TestRendererBlender(unittest.TestCase):
    # These 3D models are from ShapeNet dataset.
    # Reference images in `data/obj/[object_id]/ref` are from the dataset of LSM [Kar et al. NeurIPS 2017].
    object_ids = [
        '1a6fca5e59b00eba250a73fdbcda6406',
        '1a9e1fb2a51ffd065b07a27512172330',
        '1a8bbf2994788e2743e99e0cae970928',
        '1bdeb4aaa0aaea4b4f95630cc18536e0',
    ]
    filename_obj = './data/obj/%s/model.obj'
    filename_views = './data/obj/%s/ref/view.txt'
    filename_ref_rgba = './data/obj/%s/ref/render_%d.png'
    filename_ref_depth = './data/obj/%s/ref/depth_%d.png'
    image_h = image_w = 224

    def load_mesh(self, object_id):
        vertices, vertices_t, normals, faces, faces_t, faces_n, textures, texture_params = nmr.load_obj(
            self.filename_obj % object_id)
        # NMR is a left-handed coordinate system while Blender is right-handed.
        vertices[:, 2] *= -1

        # NMR accepts only CUDA arrays.
        vertices = torch.as_tensor(vertices).cuda()
        faces = torch.as_tensor(faces).cuda()
        vertices_t = torch.as_tensor(vertices_t).cuda()
        faces_t = torch.as_tensor(faces_t).cuda()
        textures = torch.as_tensor(textures).cuda()
        texture_params = torch.as_tensor(texture_params).cuda()

        # Normal vectors may not be contained in .obj.
        if normals is not None:
            normals = torch.as_tensor(normals).cuda()
            faces_n = torch.as_tensor(faces_n).cuda()

        # Returns nmr.Mesh.
        meshes = nmr.create_meshes(
            vertices=vertices, vertices_t=vertices_t, normals=normals, faces=faces, faces_t=faces_t,
            faces_n=faces_n, textures=textures, texture_params=texture_params)

        return meshes

    def load_camera(self, object_id, view_num=None):
        if view_num is not None:
            # load only single viewpoint
            with open(self.filename_views % object_id) as f:
                viewpoints = f.readlines()[view_num]
            azimuth, elevation, _, distance = map(float, viewpoints.split())
        else:
            # load all viewpoints
            with open(self.filename_views % object_id) as f:
                viewpoints = f.readlines()
            azimuth = np.array([float(line.split()[0]) for line in viewpoints], np.float32)
            elevation = np.array([float(line.split()[1]) for line in viewpoints], np.float32)
            distance = np.array([float(line.split()[3]) for line in viewpoints], np.float32)

        # Angles are given as degrees.
        azimuth = np.radians(azimuth + 90)  # X and Z is swapped.
        elevation = np.radians(elevation)

        # NMR accepts only CUDA arrays.
        azimuth = torch.as_tensor(azimuth, dtype=torch.float32).cuda()
        elevation = torch.as_tensor(elevation, dtype=torch.float32).cuda()
        distance = torch.as_tensor(distance, dtype=torch.float32).cuda()

        # Create viewpoints and directions from angles.
        viewpoints = nmr.compute_viewpoints(azimuth, elevation, distance)

        # Returns nmr.Camera.
        viewing_angle = math.atan(16. / 60.) * 2
        extrinsic_parameters = nmr.create_extrinsic_camera_parameters_by_looking_at(viewpoints)
        intrinsic_parameters = nmr.create_intrinsic_camera_parameters_by_viewing_angles(
            viewing_angle, viewing_angle, self.image_h, self.image_w, device=viewpoints.device)
        cameras = nmr.create_cameras(extrinsic_parameters, intrinsic_parameters)

        return cameras

    def test_mask_depth(self):
        """Test whether a rendered foreground and depth maps by NMR match these by Blender."""
        renderer = nmr.Renderer(self.image_h, self.image_w)
        backgrounds = nmr.Backgrounds()
        for oid in tqdm.tqdm(self.object_ids):
            meshes = self.load_mesh(oid)

            for view_num in tqdm.tqdm(range(0, 20)):
                cameras = self.load_camera(oid, view_num)

                images = renderer(meshes, cameras, None, backgrounds)  # [height, width, (RGBAD)]
                alpha_map = images[:, :, 3].cpu().numpy()
                depth_map = images[:, :, 4].cpu().numpy()

                # Assertion of alpha map.
                # There can be small differences, but they must not be greater than one.
                # (There must not be a difference in the presence or absence of an object.)
                ref_rgba = skimage.io.imread(self.filename_ref_rgba % (oid, view_num)).astype(np.float32) / 255
                ref_alpha = ref_rgba[:, :, 3]
                diff_alpha = ref_alpha - alpha_map
                num_diff_pixels = (np.abs(diff_alpha) == 1).sum()
                assert num_diff_pixels == 0

                # Assertion of depth map.
                # Difference between depth maps should be small (0.01) at least 90% of pixels.
                diff_threshold = 0.01
                diff_max_ratio = 0.1
                ref_depth = skimage.io.imread(self.filename_ref_depth % (oid, view_num)).astype(np.float32)
                ref_depth[ref_depth == 65535] = 0
                ref_depth = ref_depth / 65535. * 20
                diff_depth = ref_depth - depth_map
                diff_ratio = (diff_threshold < diff_depth[ref_alpha == 1]).mean()
                assert diff_ratio < diff_max_ratio

    def test_mask_depth_batch(self):
        """Test whether a rendered foreground and depth maps by NMR match these by Blender."""
        renderer = nmr.Renderer(self.image_h, self.image_w)
        backgrounds = nmr.Backgrounds()
        for oid in tqdm.tqdm(self.object_ids):
            # Render images from .obj
            meshes = self.load_mesh(oid)
            cameras = self.load_camera(oid)
            images = renderer(meshes, cameras, None, backgrounds)  # [height, width, (RGBAD)]
            alpha_maps = images[:, :, :, 3].cpu().numpy()
            depth_maps = images[:, :, :, 4].cpu().numpy()

            # Assertion of alpha map.
            ref_alpha_list = []
            for view_num in range(20):
                ref_rgba = skimage.io.imread(self.filename_ref_rgba % (oid, view_num)).astype(np.float32) / 255
                ref_alpha = ref_rgba[:, :, 3]
                ref_alpha_list.append(ref_alpha)
            ref_alpha = np.array(ref_alpha_list)
            diff_alpha = ref_alpha - alpha_maps
            num_diff_pixels = (np.abs(diff_alpha) == 1).sum()
            assert num_diff_pixels == 0

            # Assertion of depth map.
            diff_threshold = 0.01
            diff_max_ratio = 0.1
            ref_depth_list = []
            for view_num in range(20):
                ref_depth = skimage.io.imread(self.filename_ref_depth % (oid, view_num)).astype(np.float32)
                ref_depth[ref_depth == 65535] = 0
                ref_depth = ref_depth / 65535. * 20
                ref_depth_list.append(ref_depth)
            ref_depth = np.array(ref_depth_list)
            diff_depth = ref_depth - depth_maps
            diff_ratio = (diff_threshold < diff_depth[ref_alpha == 1]).mean()
            assert diff_ratio < diff_max_ratio

    def test_rgb(self):
        """Quantitative evaluation of rendered images. Outputs have to be checked by humans."""
        output_directory = '/tmp'
        view_num = 9

        for oid in tqdm.tqdm(self.object_ids):
            meshes = self.load_mesh(oid)
            cameras = self.load_camera(oid, view_num)
            backgrounds = nmr.Backgrounds()

            renderer = nmr.Renderer(self.image_h, self.image_w)
            images = renderer(meshes, cameras, None, backgrounds)

            images = (images[:, :, :4]).cpu().numpy()
            images = np.clip((images * 255), 0, 255).astype('uint8')
            skimage.io.imsave(os.path.join(output_directory, '%s_nmr.png' % oid), images)

            images_b = skimage.io.imread(self.filename_ref_rgba % (oid, view_num))
            skimage.io.imsave(os.path.join(output_directory, '%s_blender.png' % oid), images_b)


if __name__ == '__main__':
    unittest.main()
