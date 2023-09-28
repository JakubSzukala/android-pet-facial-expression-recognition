# TODO: Make this into package

import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
import albumentations as A
from tqdm import tqdm
from PIL import Image

from typing import (
    Callable,
    Optional,
    Tuple,
    List,
    Union,
    Literal
)

# Credit: https://kozodoi.me/blog/20210308/compute-image-stats
def calculate_mean_and_std(
    img_paths: Union[Tuple, List],
    transforms: Callable,
    device: Optional[Literal['cpu', 'cuda']]='cpu'
) -> Tuple[float, float]:
    """
    Function to calculate mean and standard deviation from image dataset.

    Args:
        img_paths: Iterable with paths to images.
        transforms: Transforms like resizing, normalizing and to tensor. Ones like would be used in test/val sets,
            NOT colorchanging, cropping etc, anything that would change mean/std. Recommended example:
            >>> transform = A.Compose([
                    A.Resize(height = image_size, width = image_size),
                    A.Normalize(mean = (0, 0, 0), std  = (1, 1, 1)),
                ])
        device: Device on which calculations should be performed.
    Returns:
        mean: mean of normalized data. Returned in RGB format.
        std: standard deviation of normalized data. Returned in RGB format.
    """
    device = torch.device(device)
    px_sum =   torch.tensor([0.0, 0.0, 0.0], device=device)
    px_sqsum = torch.tensor([0.0, 0.0, 0.0], device=device)

    for img_path in tqdm(img_paths):
        image = np.array(Image.open(img_path).convert('RGB'))
        image = torch.from_numpy(transforms(image=image)['image']).permute(2, 0, 1)
        px_sum += torch.sum(image, dim=(1, 2))
        px_sqsum += torch.sum(image ** 2, dim=(1, 2))

    # TODO: This is not ideal to use last member from list iteration
    channel_px_count = len(img_paths) * image.shape[1] * image.shape[2]

    # Mean and std
    total_mean = px_sum / channel_px_count
    total_var = (px_sqsum / channel_px_count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    return total_mean, total_std