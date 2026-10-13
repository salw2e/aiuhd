import numpy as np
import torch
from PIL import Image
from aiuhd import models, seedvr


class Nearest:
    scale = 2

    def __call__(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def test_tiles_stitch_to_the_same_picture():
    models.loaded["fake", "cpu"] = Nearest()
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (37, 53, 4), dtype=np.uint8), "RGBA")
    whole = models.upscale(im, "fake", tile=1000)
    tiled = models.upscale(im, "fake", tile=16, pad=3)
    assert whole.size == (106, 74) and whole.mode == "RGBA"
    assert np.array_equal(np.asarray(whole), np.asarray(tiled))
    assert np.array_equal(np.asarray(whole)[::2, ::2, :3], np.asarray(im)[:, :, :3])


def test_seedvr_tiles_cover_the_picture_and_join_back():
    for size in (300, 544, 545, 1450, 2576):
        at = seedvr.spots(size, min(size, 544))
        assert at[0] == 0 and at[-1] + min(size, 544) == size
        assert all(b - a <= 544 - seedvr.LAP for a, b in zip(at, at[1:]))
    im = Image.fromarray(np.random.default_rng(1).integers(0, 255, (700, 900, 3), dtype=np.uint8))
    tw, th = 544, 608
    tiles = {(x, y): im.crop((x, y, x + tw, y + th)).resize((tw * 2, th * 2), Image.NEAREST)
             for y in seedvr.spots(700, th) for x in seedvr.spots(900, tw)}
    joined = seedvr.join(tiles, im.size, (tw, th))
    assert np.array_equal(np.asarray(joined), np.asarray(im.resize((1800, 1400), Image.NEAREST)))
