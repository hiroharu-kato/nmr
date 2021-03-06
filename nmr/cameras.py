import torch

from . import utils


def compute_viewpoints(azimuth, elevation, distance):
    utils.assert_shape(azimuth, (None,))
    utils.assert_shape(elevation, (None,))
    utils.assert_shape(distance, (None,))
    x = distance * torch.cos(elevation) * torch.sin(azimuth)
    y = distance * torch.sin(elevation)
    z = -distance * torch.cos(elevation) * torch.cos(azimuth)
    x, y, z = torch.broadcast_tensors(x, y, z)
    viewpoints = torch.stack((x, y, z), axis=1)
    return viewpoints


def create_extrinsic_camera_parameters_by_looking_at(origin, at=None, up=None):
    # Viewpoints (origin) must be [batch_size, 3].
    utils.assert_shape(origin, (None, 3))

    # Get GPU information
    device = origin.device

    # Points to be looked at (at) must be [batch_size, 3].
    if at is None:
        at = torch.as_tensor([0, 0, 0], dtype=torch.float32, device=device)
        at = at.unsqueeze(0)
    else:
        utils.assert_shape(at, (None, 3))

    # Upper direction (up) must be [batch_size, 3].
    if up is None:
        up = torch.as_tensor([0, 1, 0], dtype=torch.float32, device=device)
        up = up.unsqueeze(0)
    else:
        utils.assert_shape(up, (None, 3))

    origin, at, up = torch.broadcast_tensors(origin, at, up)
    z = utils.normalize(at - origin, axis=-1)
    x = utils.normalize(torch.cross(up, z), axis=-1)
    y = utils.normalize(torch.cross(z, x), axis=-1)
    rotation_matrices = torch.stack((x, y, z), axis=1)
    translations = torch.matmul(rotation_matrices, -origin[:, :, None])

    extrinsic_camera_parameters = torch.cat((rotation_matrices, translations), axis=2)

    return extrinsic_camera_parameters


def create_intrinsic_camera_parameters_by_viewing_angles(viewing_angles_x, viewing_angles_y, image_h, image_w):
    utils.assert_shape(viewing_angles_x, (None,))
    utils.assert_shape(viewing_angles_y, (None,))
    device = viewing_angles_x.device

    fx = 1 / torch.tan(viewing_angles_x / 2) * min(image_h, image_w) / 2
    fy = 1 / torch.tan(viewing_angles_y / 2) * min(image_h, image_w) / 2
    tx = image_w / 2
    ty = image_h / 2
    intrinsic_matrices_base = torch.as_tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device)
    intrinsic_matrices_fx = torch.as_tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device)
    intrinsic_matrices_fy = torch.as_tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=device)
    intrinsic_matrices_tx = torch.as_tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device)
    intrinsic_matrices_ty = torch.as_tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
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

    return intrinsic_matrices


class Cameras(object):
    def __init__(self, extrinsic_parameters, intrinsic_parameters):
        utils.assert_shape(extrinsic_parameters, (None, 3, 4))
        utils.assert_shape(intrinsic_parameters, (None, 3, 3))
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
        vertices_z = vertices[:, :, 2]
        vertices_x = vertices[:, :, 0] / vertices_z
        vertices_y = vertices[:, :, 1] / vertices_z
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
