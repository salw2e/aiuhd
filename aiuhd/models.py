import os, sys, urllib.request
import numpy as np
import torch
from PIL import Image

HOME = os.path.expanduser(os.environ.get("AIUHD_HOME", "~/.cache/aiuhd"))
REAL = "https://github.com/xinntao/Real-ESRGAN/releases/download/"
MODELS = {
    "realesrgan-x4plus": REAL + "v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x2plus": REAL + "v0.2.1/RealESRGAN_x2plus.pth",
    "realesr-general-x4v3": REAL + "v0.2.5.0/realesr-general-x4v3.pth",
    "bsrgan": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
    "swinir-real": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
}
FACES = {
    "gfpgan": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "restoreformer": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
}
loaded = {}


def pick_device(want="auto"):
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fetch(url):
    path = os.path.join(HOME, os.path.basename(url))
    if not os.path.isfile(path):
        os.makedirs(HOME, exist_ok=True)
        print("downloading", os.path.basename(url), file=sys.stderr)
        try:
            urllib.request.urlretrieve(url, path + ".part")
        except OSError as e:
            sys.exit(f"could not download {url}: {e}\nrun `aiuhd models` once while online")
        os.replace(path + ".part", path)
    return path


def load(name, device="cpu"):
    if (name, device) not in loaded:
        from spandrel import ModelLoader
        loaded[name, device] = ModelLoader().load_from_file(fetch((MODELS | FACES)[name])).eval().to(device)
    return loaded[name, device]


def upscale(im, name, device="cpu", tile=256, pad=16, progress=None):
    """Runs the model over the image in overlapping tiles, so memory depends on the tile and not on the image."""
    model = load(name, device)
    s = model.scale
    src = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    h, w = src.shape[2:]
    out = torch.empty(1, 3, h * s, w * s)
    spots = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile)]
    with torch.inference_mode():
        for i, (y, x) in enumerate(spots):
            top, left = max(0, y - pad), max(0, x - pad)
            big = model(src[:, :, top:min(h, y + tile + pad), left:min(w, x + tile + pad)].to(device)).cpu()
            th, tw = min(tile, h - y), min(tile, w - x)
            out[:, :, y * s:(y + th) * s, x * s:(x + tw) * s] = big[:, :, (y - top) * s:(y - top + th) * s, (x - left) * s:(x - left + tw) * s]
            if progress:
                progress(i + 1, len(spots))
    rgb = Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    if "A" in im.getbands():
        rgb.putalpha(im.getchannel("A").resize(rgb.size, Image.LANCZOS))
    return rgb


def tone(im, colour=0.3, warm=6):
    """Colour and lighting. A hazy photo has no real black, everything dark sits on a grey floor. The floor is taken off
    the dark tones only, fading out towards the middle, so hair goes black again and skin and walls keep their brightness.
    Haze also washes colour out, so colour and warmth are given back in step with how much haze there was. A photo that
    already has a true black is left as it is. The numbers were fitted to one hazy mirror photo."""
    import cv2
    lab = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    light = lab[..., 0]
    lo, mid = np.percentile(light, (1, 50))
    floor = min(lo, 96) - 12
    if floor <= 0 or mid - lo <= 8:
        return im
    hazy = min(floor / 30, 1)
    light = light - floor * np.exp(-(light - lo).clip(0) / (0.3 * (mid - lo)))
    lab[..., 0] = light
    lab[..., 1:] = 128 + (lab[..., 1:] - 128) * (1 + colour * hazy)
    lab[..., 2] += warm * hazy * (light / 128).clip(0, 1)
    out = Image.fromarray(cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    if "A" in im.getbands():
        out.putalpha(im.getchannel("A"))
    return out
