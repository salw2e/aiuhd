import argparse, os, sys, time
from PIL import Image, ImageOps
from .models import FACES, MODELS, fetch, pick_device, tone, upscale


def bar(unit):
    start = time.time()

    def show(i, n):
        left = f"{(time.time() - start) / i * (n - i) / 60:.1f} min left" if i else "starting"
        print(f"\r{i}/{n} {unit}, {left} ", end="", file=sys.stderr)
    return show


def main():
    ap = argparse.ArgumentParser(prog="aiuhd", description="Upscale and enhance photos and videos locally")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("photo", "video"):
        p = sub.add_parser(cmd)
        p.add_argument("file")
        p.add_argument("-o", "--out")
        p.add_argument("--scale", type=float, default=2, help="1 to 4, default 2")
        p.add_argument("--model", choices=["seedvr2", *MODELS], default="seedvr2" if cmd == "photo" else "realesr-general-x4v3")
        p.add_argument("--face", choices=[*FACES, "none"], default="gfpgan" if cmd == "photo" else "none")
        p.add_argument("--no-color", action="store_true", help="leave colour and lighting alone")
        p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
        p.add_argument("--tile", type=int, default=256, help="smaller uses less memory")
    sub.add_parser("models", help="download the models now, for offline use")
    a = ap.parse_args()

    if a.cmd == "models":
        from .faces import OCCLUDER, YUNET
        from .seedvr import code
        for url in [*MODELS.values(), *FACES.values(), YUNET, OCCLUDER]:
            fetch(url)
        code()
        return
    if not os.path.isfile(a.file):
        sys.exit(f"no such file: {a.file}")
    if not 1 <= a.scale <= 4:
        sys.exit("--scale must be between 1 and 4")
    device = pick_device(a.device)
    face = None if a.face == "none" else a.face
    stem, ext = os.path.splitext(a.file)

    if a.cmd == "video":
        from .video import enhance
        if a.model == "seedvr2":
            sys.exit("seedvr2 is for photos here, pick another --model for video")
        out = a.out or f"{stem}-x{a.scale:g}.mp4"
        enhance(a.file, out, a.model, a.scale, device, a.tile, face, not a.no_color, bar("frames"))
    else:
        try:
            im = ImageOps.exif_transpose(Image.open(a.file))
        except OSError:
            sys.exit(f"not an image: {a.file}")
        im = im if "A" in im.getbands() else im.convert("RGB")
        if a.model == "seedvr2":
            from .seedvr import restore as redraw
            big = redraw(im, bar("tiles"))
        else:
            big = upscale(im, a.model, device, a.tile, progress=bar("tiles"))
        size = (round(im.width * a.scale), round(im.height * a.scale))
        if big.size != size:
            big = big.resize(size, Image.LANCZOS)
        if face:
            from .faces import restore
            big, n = restore(im, big, face, device)
            print(f"\n{n} faces restored", end="", file=sys.stderr)
        if not a.no_color:
            big = tone(big)
        out = a.out or f"{stem}-x{a.scale:g}{ext if ext.lower() in ('.png', '.jpg', '.jpeg', '.webp') else '.png'}"
        if out.lower().endswith((".jpg", ".jpeg")):
            big = big.convert("RGB")
        big.save(out, quality=95)
    print("\n" + out, file=sys.stderr)


if __name__ == "__main__":
    main()
