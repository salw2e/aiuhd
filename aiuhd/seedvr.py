import glob, math, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image
from .models import HOME

REPO, PIN = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler", "4490bd1f482e026674543386bb2a4d176da245b9"
WEIGHTS = "seedvr2_ema_3b-Q4_K_M.gguf"
TW, TH, LAP = 544, 608, 64     # a tile this size peaks at about 8.5 GB of RAM


def code():
    path = os.path.join(HOME, "seedvr2")
    if not os.path.isfile(os.path.join(path, "inference_cli.py")):
        print("downloading seedvr2", file=sys.stderr)
        subprocess.run(["git", "clone", "-q", REPO, path], check=True)
        subprocess.run(["git", "-C", path, "checkout", "-q", PIN], check=True)
        # without a gpu its loader asks safetensors for the device "cpu:0", which safetensors refuses
        f = os.path.join(path, "src", "core", "model_loader.py")
        s = open(f).read().replace("load_safetensors_file(checkpoint_path, device=device_str)",
                                   'load_safetensors_file(checkpoint_path, device="cpu" if device_str.startswith("cpu") else device_str)')
        open(f, "w").write(s)
    return path


def spots(size, tile):
    n = math.ceil((size - LAP) / (tile - LAP)) if size > tile else 1
    return [round(i * (size - tile) / max(n - 1, 1)) for i in range(n)]


def join(tiles, size, tile, s=2, feather=80):
    """tiles: {(x, y): image}. Overlaps are cross-faded so no seam shows."""
    (W, H), (tw, th) = size, tile
    acc, weight = np.zeros((H * s, W * s, 3), np.float32), np.zeros((H * s, W * s, 1), np.float32)
    ramp = lambda n: np.minimum(1, np.minimum(np.arange(n) + 1, n - np.arange(n)) / feather)
    w = (ramp(th * s)[:, None] * ramp(tw * s)[None, :])[..., None]
    for (x, y), t in tiles.items():
        t = np.asarray(t.convert("RGB").resize((tw * s, th * s), Image.LANCZOS), dtype=np.float32)
        acc[y * s:(y + th) * s, x * s:(x + tw) * s] += t * w
        weight[y * s:(y + th) * s, x * s:(x + tw) * s] += w
    return Image.fromarray((acc / weight).clip(0, 255).round().astype(np.uint8))


def restore(im, progress=None):
    """2x with SeedVR2, which redraws lost detail. Runs as its own program, one overlapping tile at a time."""
    rgb = im.convert("RGB")
    W, H = rgb.size
    tw, th = min(W, TW), min(H, TH)
    where = [(x, y) for y in spots(H, th) for x in spots(W, tw)]
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/in")
        for x, y in where:
            rgb.crop((x, y, x + tw, y + th)).save(f"{tmp}/in/t_{y:05d}_{x:05d}.png")
        run = subprocess.Popen([sys.executable, "inference_cli.py", f"{tmp}/in", "--output", f"{tmp}/out", "--dit_model", WEIGHTS,
                                "--resolution", str(min(tw, th) * 2), "--batch_size", "1", "--cache_dit", "--cache_vae"],
                               cwd=code(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        last = []
        for line in run.stdout:
            last = (last + [line])[-20:]
            if progress:
                progress(len(glob.glob(f"{tmp}/out/*")), len(where))
        if run.wait() != 0:
            sys.exit("seedvr2 failed:\n" + "".join(last))
        out = join({(x, y): Image.open(glob.glob(f"{tmp}/out/t_{y:05d}_{x:05d}*")[0]) for x, y in where}, (W, H), (tw, th))
    if "A" in im.getbands():
        out.putalpha(im.getchannel("A").resize(out.size, Image.LANCZOS))
    return out
