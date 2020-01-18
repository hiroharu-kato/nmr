import math

import torch

from . import utils


def create_extrinsic_camera_parameters_by_looking_at(origin, at=None, up=None):
    # Get GPU information
    device = origin.device

    # Viewpoints (origin) must be [3] or [batch_size, 3].
    assert origin.ndim in (1, 2)
    assert origin.shape[-1] == 3
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)

    # Points to be looked at (at) must be [3] or [batch_size, 3].
    if at is not None:
        assert at.ndim in (1, 2)
        assert at.shape[-1] == 3
    else:
        at = torch.as_tensor([0, 0, 0], dtype=torch.float32, device=device)
    if at.ndim == 1:
        at = at.unsqueeze(0)

    # Upper direction (up) must be [3] or [batch_size, 3].
    if up is not None:
        assert up.ndim in (1, 2)
        assert up.shape[-1] == 3
    else:
        up = torch.as_tensor([0, 1, 0], dtype=torch.float32, device=device)
    if up.ndim == 1:
        up = up.unsqueeze(0)

    origin, at, up = torch.broadcast_tensors(origin, at, up)
    z = utils.normalize(at - origin, axis=-1)
    x = utils.normalize(torch.cross(up, z), axis=-1)
    y = utils.normalize(torch.cross(z, x), axis=-1)
    rotation_matrices = torch.stack((x, y, z), axis=1)
    translations = torch.matmul(rotation_matrices, -origin[:, :, None])

    extrinsic_camera_parameters = torch.cat((rotation_matrices, translations), axis=2)
    extrinsic_camera_parameters = extrinsic_camera_parameters.squeeze()
    return extrinsic_camera_parameters


def create_intrinsic_camera_parameters_by_viewing_angles(viewing_angles_x, viewing_angles_y, image_h, image_w):
    if not isinstance(viewing_angles_x, torch.Tensor):
        viewing_angles_x = torch.as_tensor(viewing_angles_x, dtype=torch.float32).cuda()
    if not isinstance(viewing_angles_y, torch.Tensor):
        viewing_angles_y = torch.as_tensor(viewing_angles_y, dtype=torch.float32).cuda()
    if viewing_angles_x.ndim == 0:
        viewing_angles_x = viewing_angles_x.unsqueeze(0)
    if viewing_angles_y.ndim == 0:
        viewing_angles_x = viewing_angles_y.unsqueeze(0)
    fx = 1 / torch.tan(viewing_angles_x / 2) * image_w / 2
    fy = 1 / torch.tan(viewing_angles_y / 2) * image_h / 2
    tx = image_w / 2
    ty = image_h / 2
    intrinsic_matrices_base = torch.as_tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float32).cuda()
    intrinsic_matrices_fx = torch.as_tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).cuda()
    intrinsic_matrices_fy = torch.as_tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).cuda()
    intrinsic_matrices_tx = torch.as_tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).cuda()
    intrinsic_matrices_ty = torch.as_tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32).cuda()
    intrinsic_matrices_base = intrinsic_matrices_base.unsqueeze(0)
    intrinsic_matrices_fx = intrinsic_matrices_fx.unsqueeze(0)
    intrinsic_matrices_fy = intrinsic_matrices_fy.unsqueeze(0)
    intrinsic_matrices_tx = intrinsic_matrices_tx.unsqueeze(0)
    intrinsic_matrices_ty = intrinsic_matrices_ty.unsqueeze(0)
    intrinsic_matrices = (
            intrinsic_matrices_fx * fx +
            intrinsic_matrices_fy * fy +
            intrinsic_matrices_tx * tx +
            intrinsic_matrices_ty * ty +
            intrinsic_matrices_base)
    intrinsic_matrices = intrinsic_matrices.squeeze()
    return intrinsic_matrices


class Cameras(object):
    def __init__(self, extrinsic_parameters, intrinsic_parameters):
        assert extrinsic_parameters.ndim in (2, 3)
        assert extrinsic_parameters.shape[-2:] == (3, 4)
        if extrinsic_parameters.ndim == 2:
            extrinsic_parameters = extrinsic_parameters.unsqueeze(0)
        assert intrinsic_parameters.ndim in (2, 3)
        assert intrinsic_parameters.shape[-2:] == (3, 3)
        if intrinsic_parameters.ndim == 2:
            intrinsic_parameters = intrinsic_parameters.unsqueeze(0)

        self.extrinsic_parameters = extrinsic_parameters
        self.intrinsic_parameters = intrinsic_parameters

    def convert_to_camera_coordinates(self, vertices):
        ones = torch.ones((vertices.shape[0], vertices.shape[1], 1), dtype=torch.float32, device=vertices.device)
        vertices = torch.cat((vertices, ones), axis=2)
        vertices = torch.matmul(vertices, self.extrinsic_parameters.permute(0, 2, 1))
        return vertices

    def convert_to_screen_coordinates(self, vertices):
        intrinsic_parameters = self.intrinsic_parameters
        vertices = torch.matmul(vertices, intrinsic_parameters.permute((0, 2, 1)))
        vertices_x = vertices[:, :, 0] / vertices[:, :, 2]
        vertices_y = vertices[:, :, 1] / vertices[:, :, 2]
        vertices_z = vertices[:, :, 2]
        vertices = torch.stack((vertices_x, vertices_y, vertices_z), dim=2)
        return vertices

    def process_vertices(self, vertices):
        vertices = self.convert_to_camera_coordinates(vertices)
        vertices = self.convert_to_screen_coordinates(vertices)
        return vertices

    def process_normals(self, normals):
        rotation_matrices = self.extrinsic_parameters[:, :, :3]
        rotation_matrices = rotation_matrices.permute(0, 2, 1)
        normals = torch.matmul(normals, rotation_matrices)
        return normals


def create_cameras(extrinsic_parameters, intrinsic_parameters):
    return Cameras(extrinsic_parameters, intrinsic_parameters)
