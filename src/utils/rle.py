from __future__ import annotations
import numpy as np

def rle_decode(rle: str, height: int, width: int) -> np.ndarray:
    """Decode RLE string into binary mask (H,W). Uses Fortran order."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None or not isinstance(rle, str) or rle.strip() == "":
        return mask.reshape((height, width), order="F")

    s = rle.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape((height, width), order="F")
