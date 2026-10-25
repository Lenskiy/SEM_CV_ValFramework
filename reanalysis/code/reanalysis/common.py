"""Shared pieces of the re-analysis: data index, SEM metadata, patches, folds, model.

Follows the original notebook (SEM_CV_ValFramework) where it matters for
comparability: 3840x2550 SEM images cropped to 3840x2400 (removing the data
bar), 480 px non-overlapping patches (40 per image), three replicated grey
channels scaled to [0, 1], and the same dual-head ResNet.
"""
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PATCH = 480
CROP_H, CROP_W = 2400, 3840
CATEGORIES = ["Low", "Mid", "High", "Ultra-high"]
THRESHOLDS = (5.0, 10.0, 20.0)


def snr_category(snr):
    """0-3 with the paper's thresholds (<=5, <=10, <=20, >20)."""
    for k, t in enumerate(THRESHOLDS):
        if snr <= t:
            return k
    return 3


def sensor_key_from_folder(name):
    """'0.1M-1000uA-60s-500x' -> '0.1M-1000uA-60s' (any magnification suffix)."""
    return re.sub(r"-\d+x$", "", name)


def parse_sensor_key(key):
    """'0.1M-1000uA-60s' -> (0.1, 1000.0, 60.0)."""
    m = re.match(r"([\d.]+)M-([\d.]+)uA-([\d.]+)s$", key)
    if not m:
        raise ValueError(key)
    return tuple(float(x) for x in m.groups())


def read_sem_metadata(path):
    """Acquisition fields from the Phenom/FEI XML in TIFF tag 34683."""
    from PIL import Image
    s = str(Image.open(path).tag_v2.get(34683, ""))

    def get(tag):
        m = re.search(rf"<{tag}[^>]*>([^<]*)<", s)
        return m.group(1) if m else ""

    return dict(time=get("time"), working_distance=get("workingDistance"),
                high_voltage=get("highVoltage"), pixel_width_nm=get("pixelWidth"),
                brightness=get("appliedBrightness"), contrast=get("appliedContrast"),
                dwell_time=get("dwellTime"), emission_current=get("emissionCurrent"),
                detector=get("detector"))


@dataclass
class SemImage:
    sensor: str
    path: Path
    meta: dict = field(default_factory=dict)


def index_sem(root):
    """All TIFFs under <root>/<sensor>-500x/, sorted, with metadata."""
    images = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(folder.glob("*.tif*")):
            images.append(SemImage(sensor_key_from_folder(folder.name), f, read_sem_metadata(f)))
    return images


def load_labels(snr_csv, include_outlier=False):
    """sensor -> median SNR, from compute_snr.py output."""
    labels = {}
    with open(snr_csv) as fh:
        for r in csv.DictReader(fh):
            if r["excluded_as_outlier"] == "True" and not include_outlier:
                continue
            labels[r["sensor"]] = float(r["snr_median"])
    return labels


def load_image_array(path):
    """uint8 (2400, 3840): greyscale, data bar removed as in the notebook."""
    from PIL import Image
    im = Image.open(path).convert("L")
    w, h = im.size
    if (w, h) == (7680, 5100):          # notebook also handled 2x images
        im = im.crop((0, 0, 7680, 4800)).resize((CROP_W, CROP_H), Image.BILINEAR)
    else:
        im = im.crop((0, 0, CROP_W, CROP_H))
    return np.asarray(im, dtype=np.uint8)


def patch_grid():
    """(row, col) of the 40 patches of one image, row-major."""
    return [(r, c) for r in range(CROP_H // PATCH) for c in range(CROP_W // PATCH)]


def fab_features(sensor):
    """Encoded fabrication parameters: log10 conc, log10 current, time, log10 charge."""
    conc, cur, t = parse_sensor_key(sensor)
    return np.array([math.log10(conc), math.log10(cur), t / 60.0, math.log10(cur * t)], dtype=np.float32)


# --------------------------------------------------------------------- folds
def sensor_folds(sensors, cats, n_folds=5, seed=0):
    """Sensor-grouped folds stratified by category: list of test-sensor lists."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    by_cat = {}
    for s, c in zip(sensors, cats):
        by_cat.setdefault(c, []).append(s)
    offset = 0
    for c in sorted(by_cat):
        members = list(by_cat[c])
        rng.shuffle(members)
        for i, s in enumerate(members):
            folds[(i + offset) % n_folds].append(s)
        offset += len(members)
    return [sorted(f) for f in folds]


def image_folds(images, n_folds=4, seed=42):
    """Original notebook scheme: within each sensor, its images are split over
    the folds (KFold shuffle, random_state=42), so every sensor is in both
    train and test of every fold. Returns list of test image-index lists."""
    from sklearn.model_selection import KFold
    groups = {}
    for i, im in enumerate(images):
        groups.setdefault(im.sensor, []).append(i)
    folds = [[] for _ in range(n_folds)]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for idx in groups.values():
        for k, (_, test) in enumerate(kf.split(idx)):
            folds[k].extend(idx[t] for t in test)
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------- model
def build_model(num_classes=4, n_params=0, return_logits=True):
    """ResNetDual from the notebook (3 residual blocks, 64/128/256 filters,
    GAP, dropout 0.2, classification + regression heads).

    n_params > 0 concatenates encoded fabrication parameters to the 256-d
    features (the notebook's commented-out image + parameters variant).
    return_logits=False reproduces the original softmax-before-CE behaviour.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.shortcut = nn.Sequential()
            if stride != 1 or cin != cout:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=stride * 2, padding=1, bias=False),
                    nn.BatchNorm2d(cout))

        def forward(self, x):
            identity = self.shortcut(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + identity)

    class ResNetDual(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding="same", bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = nn.Sequential(BasicBlock(64, 64, 1))
            self.layer2 = nn.Sequential(BasicBlock(64, 128, 2))
            self.layer3 = nn.Sequential(BasicBlock(128, 256, 2))
            self.dropout = nn.Dropout(p=0.2)
            self.fcC = nn.Linear(256 + n_params, num_classes)
            self.fcR = nn.Linear(256 + n_params, 1)

        def features(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer3(self.layer2(self.layer1(out)))
            return F.adaptive_avg_pool2d(out, 1).flatten(1)

        def forward(self, x, params=None):
            f = self.features(x)
            if n_params:
                f = torch.cat([f, params], 1)
            f = self.dropout(f)
            logits = self.fcC(f)
            c = logits if return_logits else F.softmax(logits, 1)
            return c, self.fcR(f).squeeze(1)

    return ResNetDual()
