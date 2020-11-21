import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np


class LPDataset(Dataset):
    """
    Parameters
    ----------
    root: string
        Root directory of dataset where train and test folders exists.
    train: bool (optional)
        If True, creates dataset from training set, otherwise creates from
        test set.
    transform: callable (optional)
        A function/transform that takes in an PIL image and returns a
        transformed version.
    target_transform: callable (optional)
        A function/transform that takes in the target and transforms it.
    """
    json_folder = "ann"
    img_folder = "img"
    _repr_indent = 4

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root if isinstance(root, Path) else Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.usage = "train"
        else:
            self.usage = "test"

        files = []
        targets = []
        for path in (self.root / self.usage / self.json_folder).iterdir():
            if path.is_file() and path.suffix == ".json":
                files += [path.stem + ".png"]
                targets += [path.stem]
        
        self.data = pd.DataFrame({
            "file": files,
            "target": targets
        })
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # get the path to the image
        img_name = self.root / self.usage / self.img_folder / self.data.file.iloc[index]
        bgr = cv2.imread(str(img_name))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)

        if self.transform is not None:
            img = self.transform(img)

        target = self.data.target.iloc[index]
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += [f"Split: {'Train' if self.train is True else 'Test'}"]
        if self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)