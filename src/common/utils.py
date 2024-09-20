import numpy as np
import random
import torch
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image

def make_grid(images, rows, cols):
    if type(images[0]) == torch.Tensor:
        images = [transforms.ToPILImage()(image) for image in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def feature_mapping(x, B):
    # B = (num_features, in_channels)
    # x = (B, in_channels, H, W)
    # x_proj = (B, 2 * num_features, H, W)
    x_proj = torch.einsum("fc, Bchw -> Bfhw", B, (2 * np.pi * x))
    x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
    return x_proj


class RandomCenteredCrop(transforms.RandomCrop):
    """Randomly crop the input image, but instead of sampling from a uniform distribution
    to determine the crop center, this transform samples from a normal distribution centered 
    at the center of the image.
    """
    def __init__(self, size, mu=0.5, sigma=0.1):
        super().__init__(size)
        self.mu = mu
        self.sigma = sigma

    def get_params(self, img, output_size):
        """Get parameters for `crop` for a random crop centered around the image center."""
        _, h, w = TF.get_dimensions(img)
        th, tw = output_size

        # Sample from a normal distribution centered at the image center
        center_x = int(random.gauss(self.mu, self.sigma) * w)
        center_y = int(random.gauss(self.mu, self.sigma) * h)

        # Ensure the crop box is within the image bounds
        x1 = max(0, min(center_x - tw // 2, w - tw))
        y1 = max(0, min(center_y - th // 2, h - th))

        return y1, x1, th, tw