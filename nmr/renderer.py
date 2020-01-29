import torch

from . import rasterization


class Renderer(object):
    def __init__(self, image_h, image_w, near=0.1, far=1000, anti_aliasing=True):
        self.image_h = image_h
        self.image_w = image_w
        self.near = near
        self.far = far
        self.anti_aliasing = anti_aliasing

    def __call__(self, meshes, cameras, lights, backgrounds, types=('rgb', 'alpha', 'depth', 'normal')):
        vertices_w = meshes.vertices
        normals_w = meshes.normals
        vertices_t = meshes.vertices_t
        faces = meshes.faces
        faces_n = meshes.faces_n
        faces_t = meshes.faces_t
        textures = meshes.textures
        texture_params = meshes.texture_params
        image_h = self.image_h
        image_w = self.image_w

        # Project vertices from world coordinates into screen coordinates
        vertices_s = cameras.process_vertices(vertices_w)

        # Create 2x larger images to reduce aliasing
        if self.anti_aliasing:
            vertices_s = vertices_s * 2
            image_h = image_h * 2
            image_w = image_w * 2

        # face_index_maps: [batch_size, image_h, image_w]
        # foreground_maps: [batch_size, image_h, image_w]
        # non-differentiable
        face_index_maps, foreground_maps = rasterization.compute_face_index_maps(
            vertices_s, faces, image_h, image_w, self.near, self.far)

        # vertex_index_maps: [batch_size, image_h, image_w, 3]
        # differentiable w.r.t faces
        vertex_index_maps = rasterization.distribute(
            faces, face_index_maps, foreground_maps, False, True, -1)

        # vertex_maps: [batch_size, image_h, image_w, (p0, p1, p2), (x, y, z)]
        # differentiable w.r.t vertices
        vertex_maps = rasterization.distribute(
            vertices_s, vertex_index_maps, foreground_maps, True, True, default_value=0.)

        # weight_maps: [batch_size, image_h, image_w, (p0, p1, p2)]
        # differentiable w.r.t vertex_maps
        weight_maps = rasterization.compute_weight_map(vertex_maps, foreground_maps)

        # depth_maps: [batch_size, image_h, image_w]
        # differentiable w.r.t vertex_maps and weight_maps
        depth_maps = rasterization.compute_depth_maps(vertex_maps, weight_maps, foreground_maps)

        if faces_n is not None:
            # vertex_n_index_maps: [batch_size, image_h, image_w, 3]
            # differentiable w.r.t faces_n
            vertex_n_index_maps = rasterization.distribute(
                faces_n, face_index_maps, foreground_maps, False, True, -1)

            # vertex_n_w_maps: [batch_size, image_h, image_w, (p0, p1, p2), (x, y, z)]
            # differentiable w.r.t normals_w
            vertex_n_w_maps = rasterization.distribute(
                normals_w, vertex_n_index_maps, foreground_maps, True, True, default_value=0.)

            # world coordinates to camera coordinates
            normals_c = cameras.process_normals(normals_w)

            # vertex_n_c_maps: [batch_size, image_h, image_w, (p0, p1, p2), (x, y, z)]
            # differentiable w.r.t normals_c
            vertex_n_c_maps = rasterization.distribute(
                normals_c, vertex_n_index_maps, foreground_maps, True, True, default_value=0.)

            # normal_w_maps, normal_c_maps: [batch_size, image_h, image_w, 3]
            # differentiable w.r.t vertex_n_w_maps, vertex_n_c_maps, weight_maps
            normal_w_maps, normal_c_maps = rasterization.compute_normal_maps(
                vertex_n_w_maps, vertex_n_c_maps, vertex_maps, weight_maps, foreground_maps)
        else:
            normals_w = rasterization.compute_normals(vertices_w, faces)
            normals_c = cameras.process_normals(normals_w)
            normal_w_maps = rasterization.distribute(
                normals_w, face_index_maps, foreground_maps, True, True, default_value=0.)
            normal_c_maps = rasterization.distribute(
                normals_c, face_index_maps, foreground_maps, True, True, default_value=0.)
            normal_w_maps, normal_c_maps = rasterization.compute_normal_maps_no_weight(
                normal_w_maps, normal_c_maps, foreground_maps)

        # vertex_n_index_maps: [batch_size, image_h, image_w, 2]
        # differentiable w.r.t faces_t
        vertex_t_index_maps = rasterization.distribute(
            faces_t, face_index_maps, foreground_maps, False, True, -1)

        # vertex_n_w_maps: [batch_size, image_h, image_w, (p0, p1, p2), (x, y, z)]
        # differentiable w.r.t vertices_t, weight_maps
        vertex_t_maps = rasterization.distribute(
            vertices_t, vertex_t_index_maps, foreground_maps, True, True, default_value=0.)
        vertex_t_maps = rasterization.interpolate(vertex_t_maps, vertex_maps, weight_maps)
        vertex_t_maps = rasterization.mask(vertex_t_maps, foreground_maps)

        # texture_params_maps: [batch_size, image_h, image_w, 3]
        # differentiable w.r.t texture_params
        texture_params_maps = rasterization.distribute(
            texture_params, face_index_maps, foreground_maps, False, True, default_value=0)

        # color_maps: [batch_size, image_h, image_w, 3]
        # differentiable w.r.t vertex_t_maps, textures, texture_params_maps
        color_maps = rasterization.compute_color_maps(
            vertex_t_maps, textures, texture_params_maps, foreground_maps)

        #
        reflectance_maps = rasterization.reflectance_maps(normal_w_maps, normal_c_maps)

        #
        coordinate_maps = (vertex_maps * weight_maps[:, :, :, :, None]).sum(-2)

        #
        rgb_maps = color_maps * reflectance_maps.unsqueeze(-1)
        rgb_maps = rasterization.mask(rgb_maps, foreground_maps)
        if self.anti_aliasing:
            rgb_maps = rasterization.downsample(rgb_maps, foreground_maps)
            depth_maps = rasterization.downsample(depth_maps, foreground_maps)
            normal_maps = rasterization.downsample(normal_c_maps, foreground_maps)
            alpha_maps = rasterization.downsample(foreground_maps, None)
        else:
            alpha_maps = foreground_maps
        alpha_maps = 1. * alpha_maps

        images = torch.cat((rgb_maps, alpha_maps.unsqueeze(-1), depth_maps.unsqueeze(-1), normal_maps), dim=-1)
        images = images.permute(0, 3, 1, 2)

        return images
