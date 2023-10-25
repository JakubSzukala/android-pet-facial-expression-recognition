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
        # Sanity checks
        required_columns = set(['image_name', 'class_id', 'class_name', 'fixed_idx'])
        assert required_columns.issubset(df.columns)
        assert len(df) == len(df.image_name.unique())
        assert len(df) == len(df.fixed_idx.unique())

        self.df = df

        # This fixes image name to idx, safer if df rows may be modified
        self.image_index_to_image_name = {
            idx: image_name
            for idx, image_name in enumerate(df.image_name.unique())
        }
        self.image_name_to_image_index = {
            v: k for k, v in self.image_index_to_image_name.items()
        }

        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df[self.df.image_name == self.image_index_to_image_name[index]]
        image_name = row.image_name.values[0]
        label = row.class_id.values[0]
        filepath = os.path.join(self.images_dir, image_name)
        image = Image.open(filepath).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        return (index, image, label)