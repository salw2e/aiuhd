import json, shutil, subprocess, sys
from PIL import Image
from .models import tone, upscale


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate:format=duration",
                          "-of", "json", path], capture_output=True, text=True, timeout=60)
    d = json.loads(out.stdout or "{}")
    if not d.get("streams"):
        sys.exit(f"no video stream in {path}")
    s = d["streams"][0]
    rate = s.get("r_frame_rate") or "30/1"
    num, den = rate.split("/")
    turn = 0
    for side in s.get("side_data_list", []):
        turn = int(side.get("rotation", turn))
    turn = int((s.get("tags") or {}).get("rotate", turn))
    w, h = int(s["width"]), int(s["height"])
    if abs(turn) % 180 == 90:
        w, h = h, w
    return w, h, rate, round(float(d.get("format", {}).get("duration") or 0) * float(num) / float(den or 1))


def enhance(src, out, name, scale, device="cpu", tile=256, face=None, colour=True, progress=None):
    """One frame in memory at a time. Audio is copied as it is, frame rate and frame count stay the same."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg not found, install it first (apt install ffmpeg / brew install ffmpeg)")
    w, h, rate, total = probe(src)
    W, H = round(w * scale) // 2 * 2, round(h * scale) // 2 * 2
    # ponytail: variable frame rate input is resampled to its nominal rate
    decode = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={rate}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                              stdout=subprocess.PIPE)
    encode = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", rate,
                               "-i", "pipe:0", "-i", src, "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
                               "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out],
                              stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    done = 0
    try:
        while True:
            raw = decode.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            frame = Image.frombuffer("RGB", (w, h), raw, "raw", "RGB", 0, 1)
            big = upscale(frame, name, device, tile)
            if big.size != (W, H):
                big = big.resize((W, H), Image.LANCZOS)
            if face:
                from .faces import restore
                big = restore(frame, big, face, device)[0]
            if colour:
                big = tone(big)     # ponytail: measured per frame, a slow drift in a long shot is possible
            encode.stdin.write(big.tobytes())
            done += 1
            if progress:
                progress(done, max(total, done))
    except BrokenPipeError:
        pass
    finally:
        decode.kill()
    encode.stdin.close()
    err = encode.stderr.read().decode(errors="replace")
    if encode.wait() != 0 or not done:
        sys.exit(f"ffmpeg could not write {out}\n{err[-800:]}")
    return done
