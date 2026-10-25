"""E10: physical gap-bridging descriptors from the 500x images (no learning; all sensors).

The electrode digits run horizontally with a 10 um pitch (37 px at 0.27 um/px, FW 1.04 mm over
3840 px). Per image (info bar cropped, 1-99 percentile stretch, as for the ViT features):
  gap_loss    fraction of the electrode area where the gaps are obscured: vertical lock-in at
              the pitch (phase per 128 px column strip) in 16 px x 2-pitch windows; obscured
              where the amplitude < 0.5 x the image's 90th-percentile amplitude
  gap_contrast median lock-in amplitude (absolute, stretched intensity units): how visible the
              gaps are overall
A bright-particle fraction (white top-hat) was tried and dropped on the visual check: it also
counts the grain texture inside the gaps of good sensors, so it has no single physical meaning.
Sensor = mean over its images. Parameters fixed on visual checks of three example images
(good, overgrown dead, 10 s dead) before any correlation was computed.
Usage: python bridging.py <500x image dir> [--show sensor ...] ; writes data/derived/bridging.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage

import common as C

ROOT = Path(__file__).resolve().parents[2]
BAR = 2380            # rows below this hold the SEM info bar
PITCH = 37.0          # px
STRIP, WIN = 128, 16
LOSS_FRAC = 0.5


def load(path):
    a = C.load_image_array(path).astype(np.float32)[:BAR]
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def pitch(a):
    p = a.mean(1) - a.mean()
    f = np.abs(np.fft.rfft(p * np.hanning(len(p))))
    fr = np.fft.rfftfreq(len(p))
    band = (fr > 1 / 45) & (fr < 1 / 30)
    return 1 / fr[band][np.argmax(f[band])]


def amplitude(a, per):
    """Local amplitude of the vertical modulation at the digit pitch (lock-in)."""
    y = np.arange(a.shape[0])[:, None]
    w = np.exp(-2j * np.pi * y / per)
    amp = np.zeros_like(a)
    for x0 in range(0, a.shape[1], STRIP):
        s = a[:, x0:x0 + STRIP]
        s = s - ndimage.uniform_filter1d(s, int(round(per)), axis=0)          # remove slow shading
        z = s * w                                                              # demodulate
        z = ndimage.uniform_filter(z.real, (int(2 * per), WIN)) + 1j * ndimage.uniform_filter(z.imag, (int(2 * per), WIN))
        amp[:, x0:x0 + STRIP] = 2 * np.abs(z)
    return amp


def describe(a):
    per = pitch(a)
    amp = amplitude(a, per)
    m = int(2 * per)
    core = amp[m:-m, WIN:-WIN]
    lost = core < LOSS_FRAC * np.percentile(core, 90)
    return dict(pitch_px=per, gap_loss=float(lost.mean()), gap_contrast=float(np.median(core))), lost


def show(sem, sensors, out):
    from PIL import Image
    for im in C.index_sem(sem):
        if im.sensor not in sensors:
            continue
        a = load(im.path)
        d, lost = describe(a)
        rgb = np.stack([a, a, a], -1)
        m = int(2 * d["pitch_px"])
        L = np.zeros(a.shape, bool)
        L[m:-m, WIN:-WIN] = lost
        rgb[L] = 0.5 * rgb[L] + 0.5 * np.array([0.92, 0.41, 0.2])
        Image.fromarray((rgb[:1200, :1800] * 255).astype(np.uint8)).save(out / f"bridging_{im.sensor}.png")
        print(im.sensor, d)
        sensors = [s for s in sensors if s != im.sensor]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sem")
    ap.add_argument("--show", nargs="*")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.show:
        return show(a.sem, a.show, Path(a.out))
    per_sensor = {}
    for im in C.index_sem(a.sem):
        d, _ = describe(load(im.path))
        per_sensor.setdefault(im.sensor, []).append(d)
    with open(ROOT / "data/derived/bridging.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "n_images", "pitch_px", "gap_loss", "gap_contrast"])
        for s, ds in sorted(per_sensor.items()):
            w.writerow([s, len(ds)] + [round(float(np.mean([d[k] for d in ds])), 5) for k in ("pitch_px", "gap_loss", "gap_contrast")])
    print("wrote", len(per_sensor), "sensors")


if __name__ == "__main__":
    main()
