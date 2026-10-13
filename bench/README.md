# Model test

100 photos from [AIM-500](https://github.com/JizhiziLi/AIM), shrunk 4x (clean, and saved as a quality 40 JPEG), upscaled back and compared with the original using LPIPS. Lower is better. Seconds are per 256 px photo on 8 threads of a Ryzen 9 7900, while the machine was busy.

| Model | Clean | JPEG | Seconds |
|---|---|---|---|
| swinir-real | 0.163 | 0.229 | 9.0 |
| realesrgan-x4plus | 0.172 | 0.233 | 2.9 |
| bsrgan | 0.176 | 0.230 | 3.1 |
| realesr-general-x4v3 | 0.222 | 0.285 | 0.12 |
| plain resize | 0.269 | 0.449 | 0.01 |

The top three are within noise of each other. SeedVR2 has not been scored on this set yet.

```
pip install lpips
sh bench/run.sh data        # data/image/*.jpg and data/set.json, see bgremove's bench/data.py
python bench/score.py data/out
```
