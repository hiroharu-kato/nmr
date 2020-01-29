import glob
import math
import os
import subprocess

import numpy as np
import skimage
import torch
import tqdm

import nmr


def load_teapot(filename='./data/obj/teapot.obj'):
    # Load data from .obj file.
    # Normal vectors may not be contained in .obj.
    vertices, vertices_t, normals, faces, faces_t, faces_n, textures, texture_params = nmr.load_obj(filename)

    # [x, z, y] -> [x, y, z]
    # max(y) + min(y)
    x, z, y = vertices.transpose()
    y = y - 0.5 * (y.max() + y.min())
    vertices = np.stack((x, y, z), axis=1)
    x, z, y = normals.transpose()
    normals = np.stack((x, y, z), axis=1)

    # normalize size
    vertices = vertices / np.abs(vertices).max()

    # Reshape to minibatch
    vertices = vertices[None]
    vertices_t = vertices_t[None]
    normals = normals[None]
    textures = textures[None]

    # NMR accepts only CUDA arrays.
    vertices = torch.as_tensor(vertices).cuda()
    faces = torch.as_tensor(faces).cuda()
    vertices_t = torch.as_tensor(vertices_t).cuda()
    faces_t = torch.as_tensor(faces_t).cuda()
    textures = torch.as_tensor(textures).cuda()
    texture_params = torch.as_tensor(texture_params).cuda()
    normals = torch.as_tensor(normals).cuda()
    faces_n = torch.as_tensor(faces_n).cuda()

    # Returns nmr.Mesh.
    meshes = nmr.create_meshes(
        vertices=vertices, vertices_t=vertices_t, normals=normals, faces=faces, faces_t=faces_t,
        faces_n=faces_n, textures=textures, texture_params=texture_params)

    return meshes


def run():
    image_h = 480  # Height of rendered image
    image_w = 640  # Width of rendered image
    viewing_angle = math.atan(16. / 60.) * 2  # Viewing angle in radians
    elevation = 15  # Elevation of viewpoints in degrees
    distance = 4  # Distance between object center and viewpoints

    #
    meshes = load_teapot()
    backgrounds = nmr.Backgrounds()
    renderer = nmr.Renderer(image_h, image_w)

    # Create intrinsic camera matrix
    viewing_angle = np.array(viewing_angle, np.float32)[None]
    viewing_angle = torch.as_tensor(viewing_angle, dtype=torch.float32).cuda()
    intrinsic_parameters = nmr.create_intrinsic_camera_parameters_by_viewing_angles(
        viewing_angle, viewing_angle, image_h, image_w)

    for azimuth in tqdm.tqdm(range(0, 360, 8)):
        # Create extrinsic camera matrix
        azimuth_t = np.radians(np.array(azimuth, dtype=np.float32))[None]
        elevation_t = np.radians(np.array(elevation, np.float32))[None]
        distance_t = np.array(distance, np.float32)[None]
        azimuth_t = torch.as_tensor(azimuth_t, dtype=torch.float32).cuda()
        elevation_t = torch.as_tensor(elevation_t, dtype=torch.float32).cuda()
        distance_t = torch.as_tensor(distance_t, dtype=torch.float32).cuda()
        viewpoints = nmr.compute_viewpoints(azimuth_t, elevation_t, distance_t)
        extrinsic_parameters = nmr.create_extrinsic_camera_parameters_by_looking_at(viewpoints)

        # Create camera
        cameras = nmr.create_cameras(extrinsic_parameters, intrinsic_parameters)

        # Render and save
        images = renderer(meshes, cameras, None, backgrounds)
        image = images[0, :3].cpu().numpy().transpose((1, 2, 0))
        image = (image * 255).astype(np.uint8)
        skimage.io.imsave('/tmp/_tmp_%04d.png' % int(azimuth), image)

    # Generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize -dispose 2'
    subprocess.call('convert  %s /tmp/_tmp_*.png /tmp/example1.gif' % options, shell=True)

    # Remove temporary files
    for filename in glob.glob('/tmp/_tmp_*.png'):
        os.remove(filename)


if __name__ == '__main__':
    run()
