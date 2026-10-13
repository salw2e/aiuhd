import io, json, os, resource, sys, time
import numpy as np
import torch, lpips
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiuhd.models import upscale

data, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 8)
seen = lpips.LPIPS(net="alex", verbose=False)


def t(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]


def small(gt, kind):
    lr = gt.resize((gt.width // 4, gt.height // 4), Image.BICUBIC)
    if kind == "jpeg":
        buf = io.BytesIO()
        lr.save(buf, "JPEG", quality=40)
        lr = Image.open(buf).convert("RGB")
    return lr


res = {"clean": {}, "jpeg": {}}
for n in sorted(json.load(open(f"{data}/set.json"))):
    gt = Image.open(f"{data}/image/{n}.jpg").convert("RGB")
    gt.thumbnail((1024, 1024), Image.LANCZOS)
    gt = gt.crop((0, 0, gt.width // 4 * 4, gt.height // 4 * 4))
    for kind in res:
        lr = small(gt, kind)
        t0 = time.time()
        big = lr.resize(gt.size, Image.LANCZOS) if name == "lanczos" else upscale(lr, name)
        sec = time.time() - t0
        a, b = np.asarray(big, dtype=np.float64), np.asarray(gt, dtype=np.float64)
        with torch.inference_mode():
            far = float(seen(t(big), t(gt)))
        res[kind][n] = {"psnr": float(10 * np.log10(255 ** 2 / ((a - b) ** 2).mean())), "lpips": far, "sec": sec}
        os.makedirs(f"{out}/pred/{name}/{kind}", exist_ok=True)
        big.save(f"{out}/pred/{name}/{kind}/{n}.jpg", quality=95)
        if name == "lanczos":
            os.makedirs(f"{out}/lr/{kind}", exist_ok=True)
            lr.save(f"{out}/lr/{kind}/{n}.png")
            os.makedirs(f"{out}/gt", exist_ok=True)
            gt.save(f"{out}/gt/{n}.jpg", quality=95)
    print(name, n, {k: round(res[k][n]["lpips"], 3) for k in res}, round(sec, 1), flush=True)
res["ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
json.dump(res, open(f"{out}/{name}.json", "w"))
