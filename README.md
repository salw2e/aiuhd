# aiuhd

Upscales and sharpens photos and videos on your own computer. It works without a GPU but it's slow.

```
pip install .
aiuhd photo in.jpg -o out.jpg
aiuhd video in.mp4 -o out.mp4
```

Needs Python 3.10+ and git, and ffmpeg for video. Without a GPU install the CPU build of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. Models download on first use, about 3 GB for photos.

```
aiuhd photo in.jpg --scale 4
aiuhd photo in.jpg --face none --no-color
aiuhd photo in.jpg --model realesrgan-x4plus
aiuhd video in.mp4 --model realesrgan-x2plus
aiuhd models
```

<picture>
  <source media="(min-width: 768px)" srcset="demo/pairs-wide.jpg">
  <img src="demo/pairs-tall.jpg" width="100%" alt="before and after: lion, teddy bear, plant">
</picture>

Photos go through [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) at 2x, then GFPGAN on faces that nothing is covering, then a colour fix for photos with no true black. `--face none` and `--no-color` turn those two steps off. Video uses a small Real-ESRGAN model frame by frame.

On a 12-core CPU with no GPU, a 1450x2576 photo takes about 21 minutes with the default model and 40 seconds with `--model realesr-general-x4v3`. Video runs at under 1 frame per second. Other models are compared in [bench](bench/).

The detail it adds is generated, so small text and faces can come out slightly different from the real thing.

## Licences

The code here is MIT. Models are downloaded from their authors: SeedVR2 (Apache 2.0, run through [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)), Real-ESRGAN (BSD 3), BSRGAN (MIT), SwinIR (Apache 2.0), GFPGAN (Apache 2.0, with some non-commercial third-party parts), RestoreFormer (Apache 2.0), YuNet (MIT), and the XSeg occlusion model from DeepFaceLab (GPL 3.0) via facefusion-assets.
