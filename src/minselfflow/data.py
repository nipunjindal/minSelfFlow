"""CIFAR-10 loading: download parquet from Hugging Face once, cache as a uint8 tensor.

The whole training set is 50000 x 3 x 32 x 32 bytes = 150MB, so we keep it in memory
(or on the GPU) and skip DataLoader workers entirely.
"""

import io
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import Tensor

from minselfflow.utils.log import logger

HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/%s-00000-of-00001.parquet"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_cifar10(root: str | Path = "data", split: Literal["train", "test"] = "train") -> tuple[Tensor, Tensor]:
    """Load CIFAR-10, downloading and caching on first use.

    Args:
        root (str | Path): cache directory.
        split (Literal["train", "test"]): which split.

    Returns:
        tuple[Tensor, Tensor]: images uint8 (B, 3, 32, 32) in [0, 255], labels int64 (B,).
    """
    root = Path(root)
    cache = root / ("cifar10_%s.pt" % split)
    if cache.exists():
        data = torch.load(cache)
        return data["images"], data["labels"]

    root.mkdir(parents=True, exist_ok=True)
    parquet = root / ("cifar10_%s.parquet" % split)
    if not parquet.exists():
        logger.info("downloading CIFAR-10 %s to %s", split, parquet)
        urllib.request.urlretrieve(HF_URL % split, parquet)

    table = pq.read_table(parquet).to_pydict()
    images = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in table["img"]])
    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(table["label"], dtype=torch.long)
    assert images.shape[1:] == (3, 32, 32) and len(images) == len(labels), images.shape
    torch.save({"images": images, "labels": labels}, cache)
    logger.info("cached %d images to %s", len(images), cache)
    return images, labels


def to_model_range(images: Tensor) -> Tensor:
    """uint8 [0, 255] -> float [-1, 1]."""
    return images.float() / 127.5 - 1.0


def to_image_range(x: Tensor) -> Tensor:
    """float [-1, 1] -> float [0, 1] for plotting."""
    return ((x + 1) / 2).clamp(0, 1)
