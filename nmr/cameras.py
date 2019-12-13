import math
import torch
from . import utils


class Cameras(object):
    def __init__(
            self, rotation_matrices=None, translations=None, origin=None, at=None, up=None,
            intrinsic_matrices=None, viewing_angles_y=None, viewing_angles_x=None, image_h=256, image_w=256):
        device = origin.device
        self.image_h = image_h
        self.image_w = image_w

        # rotation_matrices: [batch_size, 3, 3] or [3, 3]
        # translations: [batch_size, 3] or [3]
        if rotation_matrices is None:
            # create rotation matrices from (origin, at, up).
            # origin can be [batch_size, 3] or [3]
            if at is None:
                at = torch.as_tensor([0, 0, 0], dtype=torch.float32, device=device)
            if up is None:
                up = torch.as_tensor([0, 1, 0], dtype=torch.float32, device=device)
            if max(origin.ndim, at.ndim, up.ndim) == 2:
                if origin.ndim == 1:
                    origin = origin[None, :]
                elif at.ndim == 1:
                    at = at[None, :]
                elif up.ndim == 1:
                    up = up[None, :]
            z = utils.normalize(at - origin, axis=-1)
            x = utils.normalize(torch.cross(up, z), axis=-1)
            y = utils.normalize(torch.cross(z, x), axis=-1)
            if origin.ndim == 2:
                rotation_matrices = torch.cat((x[:, None, :], y[:, None, :], z[:, None, :]), axis=1)
            else:
                rotation_matrices = torch.cat((x[None, :], y[None, :], z[None, :]), axis=0)
            translations = origin
        self.rotation_matrices = rotation_matrices
        self.translations = -translations

        if intrinsic_matrices is None:
            if viewing_angles_x is None:
                viewing_angles_x = torch.as_tensor(math.atan(16. / 60.) * 2, dtype=torch.float32, device=device)
            if viewing_angles_y is None:
                viewing_angles_y = torch.as_tensor(math.atan(16. / 60.) * 2, dtype=torch.float32, device=device)
            fx = 1 / torch.tan(viewing_angles_x / 2) * image_w / 2
            fy = 1 / torch.tan(viewing_angles_y / 2) * image_h / 2
            tx = image_w / 2
            ty = image_h / 2
            intrinsic_matrices_base = torch.as_tensor(
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device)
            intrinsic_matrices_fx = torch.as_tensor(
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device)
            intrinsic_matrices_fy = torch.as_tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=device)
            intrinsic_matrices_tx = torch.as_tensor(
                [[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device)
            intrinsic_matrices_ty = torch.as_tensor(
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
            intrinsic_matrices = (
                    intrinsic_matrices_fx * fx +
                    intrinsic_matrices_fy * fy +
                    intrinsic_matrices_tx * tx +
                    intrinsic_matrices_ty * ty +
                    intrinsic_matrices_base
            )
        self.intrinsic_matrices = intrinsic_matrices

    def __call__(self, vertices):
        if self.translations.ndim == 2:
            vertices = vertices + self.translations
        elif self.translations.ndim == 1:
            vertices = vertices + self.translations[None, :]

        if self.rotation_matrices.ndim == 3:
            raise NotImplementedError
        elif self.rotation_matrices.ndim == 2:
            rotation_matrices = self.rotation_matrices.permute(1, 0)
            vertices = torch.matmul(vertices, rotation_matrices)

        intrinsic_matrices = self.intrinsic_matrices.permute(1, 0)
        vertices = torch.matmul(vertices, intrinsic_matrices)
        if vertices.ndim == 3:
            vertices_x = vertices[:, :, 0] / vertices[:, :, 2]
            vertices_y = vertices[:, :, 1] / vertices[:, :, 2]
            vertices_z = vertices[:, :, 2]
            vertices = torch.cat((vertices_x[:, :, None], vertices_y[:, :, None], vertices_z[:, :, None]), dim=2)
        elif vertices.ndim == 2:
            vertices_x = vertices[:, 0] / vertices[:, 2]
            vertices_y = vertices[:, 1] / vertices[:, 2]
            vertices_z = vertices[:, 2]
            vertices = torch.cat((vertices_x[:, None], vertices_y[:, None], vertices_z[:, None]), dim=1)

        return vertices


def create_cameras(
        rotation_matrices=None, translations=None, origin=None, at=None, up=None,
        intrinsic_matrices=None, viewing_angles_y=None, viewing_angles_x=None, image_h=256, image_w=256,
):
        return Cameras(
            rotation_matrices, translations, origin, at, up, intrinsic_matrices, viewing_angles_y,
            viewing_angles_x, image_h, image_w)