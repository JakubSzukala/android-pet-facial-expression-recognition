import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import albumentations as A

import matplotlib.pyplot as plt

# Typing
from pathlib import Path
from typing import Callable, Optional, List


class TrashnetDataset(Dataset):
    # For entire dataset
    RGB_MEANS = [0.6730, 0.6398, 0.6048]
    RGB_STD = [0.2089, 0.2099, 0.2316]

    class_id_to_class_name = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]

    class_name_to_class_id = {
        k: v for v, k in enumerate(class_id_to_class_name)
    }

    def __init__(
            self,
            df: pd.DataFrame,
            images_dir: Path,
            transform: Optional[Callable]=None,
        ) -> None:
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        filepath = os.path.join(
            self.images_dir,
            self.df.iloc[index, self.df.columns.get_loc('image_name')]
        )
        label = self.df.iloc[index, self.df.columns.get_loc('class_id')]
        image = Image.open(filepath).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        return (index, image, label)