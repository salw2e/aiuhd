import cv2
import numpy as np
import torch
from PIL import Image
from .models import fetch, load

YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
OCCLUDER = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_2.onnx"
nets = {}
# where the eyes, nose and mouth corners sit in the 512 px crop the face models were trained on (FFHQ alignment)
TEMPLATE = np.float32([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]])


def find(bgr, limit=1280):
    k = min(1, limit / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
    det = cv2.FaceDetectorYN.create(fetch(YUNET), "", (small.shape[1], small.shape[0]), 0.7, 0.3, 50)
    found = det.detect(small)[1]
    return [] if found is None else [f[4:14].reshape(5, 2) / k for f in found]


def visible(crop):
    """Which pixels of the 512 px crop are face with nothing in front. A phone, a hand or hair over it stays as it was."""
    if "occluder" not in nets:
        nets["occluder"] = cv2.dnn.readNetFromONNX(fetch(OCCLUDER))
    nets["occluder"].setInput(cv2.resize(crop, (256, 256)).astype(np.float32)[None] / 255)
    mask = cv2.resize(np.squeeze(nets["occluder"].forward()).clip(0, 1).astype(np.float32), (512, 512))
    return (cv2.GaussianBlur(mask, (0, 0), 5).clip(0.5, 1) - 0.5) * 2


def restore(im, big, name="gfpgan", device="cpu", least=0.5, strength=0.8, even=0.8):
    """Faces found in the original are redrawn by a face model and blended back, only where the face is in plain view."""
    model = load(name, device).model
    s = big.width / im.width
    src = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    canvas = np.asarray(big.convert("RGB"), dtype=np.float32)
    soft = np.zeros((512, 512), np.float32)
    soft[48:464, 48:464] = 1
    soft = cv2.GaussianBlur(soft, (0, 0), 16)
    count = 0
    # Faces are looked for in the original, not the upscaled picture: there the detector also fires on out of focus
    # people in the background, and the face model then paints a sharp stranger onto the blur.
    for marks in find(src):
        m = cv2.estimateAffinePartial2D(np.float32(marks * s), TEMPLATE, method=cv2.LMEDS)[0]
        # a face much bigger than the model's 512 px output would come back softer, so it is left alone
        if m is None or np.hypot(m[0, 0], m[0, 1]) < least:
            continue
        crop = cv2.warpAffine(canvas[:, :, ::-1].clip(0, 255).astype(np.uint8), m, (512, 512), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        x = torch.from_numpy(crop[:, :, ::-1].astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device)
        with torch.inference_mode():
            y = model(x)[0]
        face = ((y[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        seen = visible(crop)
        if even and (seen > 0.5).any():
            # A face under a fringe or against a window is half in shadow. The dark side is lifted towards the lit side,
            # measured on a heavy blur so pores and lashes are not flattened with it.
            lab = cv2.cvtColor(face.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            lit = np.percentile(lab[..., 0][seen > 0.5], 75)
            gain = 1 + seen * even * (lit - cv2.GaussianBlur(lab[..., 0], (0, 0), 25)).clip(0) / lab[..., 0].clip(1)
            lab[..., 0] *= gain
            lab[..., 1:] = 128 + (lab[..., 1:] - 128) * gain[..., None]     # lifted skin keeps its colour, or it goes grey
            face = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        back = cv2.invertAffineTransform(m)
        size = (big.width, big.height)
        a = cv2.warpAffine(soft * seen * strength, back, size)[..., None]
        canvas = canvas * (1 - a) + cv2.warpAffine(face, back, size, flags=cv2.INTER_CUBIC) * a
        count += 1
    out = Image.fromarray(canvas.clip(0, 255).round().astype(np.uint8))
    if "A" in big.getbands():
        out.putalpha(big.getchannel("A"))
    return out, count
