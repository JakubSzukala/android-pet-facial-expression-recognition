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
def calculate_variance_single_pass(
    img_paths: Union[Tuple, List],
    resize_to: Optional[Tuple[int, int]]=None,
    return_mean: bool=False
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Function to calculate variance from image dataset. It performs a single pass over all data.
    This is faster but may lead to numerical instability.

    Args:
        img_paths: Iterable with paths to images.
        resize_to: Tuple containing image size to resize to. None to not resize.
        return_mean: Whether to return mean calculated during the pass over the data.
    Returns:
        var: variance of normalized data. Returned in RGB format.
        mean: mean of normalized data. Returned in RGB format.
    """
    # TODO: Add different possible backend? torch / numpy
    px_sum =   torch.tensor([0.0, 0.0, 0.0])
    px_sqsum = torch.tensor([0.0, 0.0, 0.0])

    for img_path in tqdm(img_paths):
        # TODO: Encapsulate image loading in function
        image = np.array(Image.open(img_path).convert('RGB'))
        image = image / 255.0 # Normalize
        if resize_to is not None: # If requested resize
            image = A.Resize(resize_to[0], resize_to[1])(image=image)['image']
        image = torch.from_numpy(image).permute(2, 0, 1) # Follow PyTorch convention
        px_sum += torch.sum(image, dim=(1, 2))
        px_sqsum += torch.sum(image ** 2, dim=(1, 2))

    channel_px_count = len(img_paths) * image.shape[1] * image.shape[2]

    # Mean and var
    total_mean = px_sum / channel_px_count
    total_var = (px_sqsum / channel_px_count) - (total_mean ** 2)
    out = (total_var, total_mean) if return_mean else total_var
    return out


def calculate_variance_two_pass(
    img_paths: Union[Tuple, List],
    resize_to: Optional[Tuple[int, int]]=None,
    return_mean: bool=False
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Function to calculate variance from image dataset. It performs a two passesover all data.
    It should be more precise for single pass, but it will be much slower. It will also loss
    precision for large datasets.

    Args:
        img_paths: Iterable with paths to images.
        resize_to: Tuple containing image size to resize to. None to not resize.
        return_mean: Whether to return mean calculated during the pass over the data.
    Returns:
        var: variance of normalized data. Returned in RGB format.
        mean: mean of normalized data. Returned in RGB format.

    """
    # Compute mean
    px_sum = torch.tensor([0.0, 0.0, 0.0])
    for img_path in tqdm(img_paths):
        image = np.array(Image.open(img_path).convert('RGB'))
        image = image / 255.0 # Normalize
        if resize_to is not None: # If requested resize
            image = A.Resize(resize_to[0], resize_to[1])(image=image)['image']
        image = torch.from_numpy(image).permute(2, 0, 1) # Follow PyTorch convention
        px_sum += torch.sum(image, dim=(1, 2))

    channel_px_count = len(img_paths) * image.shape[1] * image.shape[2]
    total_mean = px_sum / channel_px_count

    # Compute variance
    px_centered_sqsum = torch.tensor([0.0, 0.0, 0.0])
    for img_path in tqdm(img_paths):
        image = np.array(Image.open(img_path).convert('RGB'))
        image = image / 255.0 # Normalize
        if resize_to is not None: # If requested resize
            image = A.Resize(resize_to[0], resize_to[1])(image=image)['image']
        image = torch.from_numpy(image).permute(2, 0, 1) # Follow PyTorch convention
        px_centered_sqsum += torch.sum((image - total_mean.view(3, 1, 1)) ** 2, dim=(1, 2))

    total_var = px_centered_sqsum / channel_px_count
    out = (total_var, total_mean) if return_mean else total_var
    return out



def check_imges_sizes(
    img_paths: Union[Tuple, List]
) -> np.ndarray:
    """
    Function to check image dimensionality across the dataset.

    Args:
        img_paths: Iterable with paths to images.

    Returns:
        uniques and counts: ndarray with unique image sizes and their counts.
    """
    img_sizes = []
    for img_path in tqdm(img_paths):
        image = np.array(Image.open(img_path).convert('RGB'))
        image = np.array(image)
        img_sizes.append(image.shape)

    return np.unique(img_sizes, return_counts=True)
