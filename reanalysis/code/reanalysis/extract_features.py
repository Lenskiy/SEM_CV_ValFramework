"""Frozen pretrained features for every tile of every SEM image (GPU).

Backbones: dinov2_vits14, dinov2_vitb14 (CLS token + mean patch token),
resnet50 ImageNet (global average pool), micronet_r50 / imnet_micronet_r50 (ResNet-50
pretrained on micrographs, from scratch or after ImageNet; NASA MicroNet); shallow ImageNet
CNNs: resnet18 (full), resnet18_l2, resnet50_l2, resnet50_l3 (truncated after layer2 / layer3,
global average pool). Tilings of the 2400 x 3840 crop:
  p480  the notebook's 40 tiles of 480 px, resized to 224
  n224  native-resolution 224 px tiles (10 x 17 = 170 per image)
  n448  native-resolution 448 px tiles (5 x 8 = 40), DINOv2 only
Intensity: 'raw' (0-255 scaled to 0-1) or 'pct' (per-image 1st-99th percentile
stretch, to remove brightness/contrast differences between imaging sessions).

Writes <out>/<backbone>_<tiling>_<norm>.npz (feats float16, img, row, col) and
<out>/images.csv (img index, sensor, file, acquisition time).
Labels are not read here; the locked test sensors are extracted like the rest
and only excluded at evaluation time.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C

DINO_REPO = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
MICRONET = {   # NASA pretrained-microscopy-models, ResNet-50 encoders (state dicts)
    "micronet_r50": Path.home() / ".cache/torch/micronet/resnet50_pretrained_microscopynet_v1.1.pth.tar",
    "imnet_micronet_r50": Path.home() / ".cache/torch/micronet/resnet50_pretrained_imagenet-microscopynet_v1.0.pth.tar",
}
SHALLOW = {"resnet18": ("resnet18", 0), "resnet18_l2": ("resnet18", 2),
           "resnet50_l2": ("resnet50", 2), "resnet50_l3": ("resnet50", 3)}   # (architecture, last layer kept; 0 = all)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TILINGS = {"p480": (480, 224), "n224": (224, 224), "n448": (448, 448)}   # tile size, network input


def load_backbone(name, device):
    if name.startswith("dinov2"):
        m = torch.hub.load(str(DINO_REPO), name, source="local")
    elif name in MICRONET:
        import torchvision
        m = torchvision.models.resnet50(weights=None)
        missing, unexpected = m.load_state_dict(torch.load(MICRONET[name], map_location="cpu"), strict=False)
        assert not unexpected and set(missing) <= {"fc.weight", "fc.bias"}, (missing, unexpected)
        m.fc = torch.nn.Identity()
    elif name in SHALLOW:
        import torchvision
        arch, cut = SHALLOW[name]
        m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if arch == "resnet18" else "IMAGENET1K_V2")
        if cut:
            keep = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"][:4 + cut]
            m = torch.nn.Sequential(*[getattr(m, k) for k in keep], torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        else:
            m.fc = torch.nn.Identity()
    else:
        import torchvision
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
    return m.eval().to(device)


def embed(model, name, x):
    if name.startswith("dinov2"):
        f = model.forward_features(x)
        return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)
    return model(x)


def tiles(a, size):
    rows, cols = a.shape[0] // size, a.shape[1] // size
    return [(r, c, a[r * size:(r + 1) * size, c * size:(c + 1) * size]) for r in range(rows) for c in range(cols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbones", default="dinov2_vits14,dinov2_vitb14,resnet50")
    ap.add_argument("--tilings", default=",".join(TILINGS))
    ap.add_argument("--norms", default="raw,pct")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = C.index_sem(args.sem)
    if args.limit:
        images = images[:args.limit]
    with open(out / "images.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img", "sensor", "file", "time"])
        for i, im in enumerate(images):
            w.writerow([i, im.sensor, im.path.name, im.meta.get("time", "")])
    arrays = [C.load_image_array(im.path) for im in images]
    print(f"{len(images)} images loaded", flush=True)

    for bb in args.backbones.split(","):
        model = load_backbone(bb, device)
        for tiling in args.tilings.split(","):
            size, inp = TILINGS[tiling]
            if tiling == "n448" and bb == "resnet50":
                continue
            for norm in args.norms.split(","):
                feats, meta = [], []
                buf, bmeta = [], []

                def flush():
                    x = torch.from_numpy(np.stack(buf)).float().unsqueeze(1)
                    if inp != size:
                        x = F.interpolate(x, size=(inp, inp), mode="bilinear", antialias=True, align_corners=False)
                    x = ((x.expand(-1, 3, -1, -1) - MEAN) / STD).to(device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        feats.append(embed(model, bb, x).float().cpu().numpy().astype(np.float16))
                    meta.extend(bmeta)
                    buf.clear()
                    bmeta.clear()

                for i, a in enumerate(arrays):
                    f = a.astype(np.float32)
                    if norm == "pct":
                        lo, hi = np.percentile(f, [1, 99])
                        f = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
                    else:
                        f = f / 255.0
                    for r, c, t in tiles(f, size):
                        buf.append(t)
                        bmeta.append((i, r, c))
                        if len(buf) == args.batch:
                            flush()
                if buf:
                    flush()
                m = np.array(meta)
                np.savez(out / f"{bb}_{tiling}_{norm}.npz", feats=np.concatenate(feats),
                         img=m[:, 0], row=m[:, 1], col=m[:, 2])
                print(f"{bb} {tiling} {norm}: {len(m)} tiles x {feats[0].shape[1]} dims", flush=True)
        del model
        torch.cuda.empty_cache()
    print("RESULT features done")


if __name__ == "__main__":
    main()
