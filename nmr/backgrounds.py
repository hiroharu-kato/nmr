import torch


class Backgrounds(object):
    def __init__(self, background_images=None, image_h=256, image_w=256):
        if background_images is None:
            self.backgrounds = torch.zeros((image_h, image_w), dtype=torch.float32).cuda()

    def __call__(self):
        return self.backgrounds


def create_backgrounds(background_images=None, image_h=256, image_w=256):
    return Backgrounds(background_images, image_h, image_w)
