import json, random, sys

out = sys.argv[1]
NAMES = ["lanczos", "realesr-general-x4v3", "realesrgan-x4plus", "bsrgan", "swinir-real"]
res = {m: json.load(open(f"{out}/{m}.json")) for m in NAMES}
photos = sorted(res["lanczos"]["clean"])


def mean(m, kind, key):
    return sum(res[m][kind][n][key] for n in photos) / len(photos)


def median(m, key):
    v = sorted(res[m][k][n][key] for k in ("clean", "jpeg") for n in photos)
    return v[len(v) // 2]


print("| model | lpips clean | lpips jpeg | psnr clean | psnr jpeg | sec | ram GB |\n|---|---|---|---|---|---|---|")
for m in sorted(NAMES, key=lambda m: mean(m, "jpeg", "lpips")):
    print(f"| {m} | {mean(m, 'clean', 'lpips'):.3f} | {mean(m, 'jpeg', 'lpips'):.3f} | {mean(m, 'clean', 'psnr'):.1f} | "
          f"{mean(m, 'jpeg', 'psnr'):.1f} | {median(m, 'sec'):.2f} | {res[m]['ram_gb']:.1f} |")

random.seed(0)
print()
for a, b in (("realesrgan-x4plus", "bsrgan"), ("realesrgan-x4plus", "swinir-real"), ("realesrgan-x4plus", "realesr-general-x4v3"),
             ("realesr-general-x4v3", "lanczos")):
    for kind in ("clean", "jpeg"):
        d = [res[a][kind][n]["lpips"] - res[b][kind][n]["lpips"] for n in photos]
        boots = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(10000))
        print(f"{a} - {b} ({kind}): {sum(d) / len(d):+.3f}  95% [{boots[250]:+.3f}, {boots[9750]:+.3f}]  better on {sum(x < 0 for x in d)}/{len(d)}")
