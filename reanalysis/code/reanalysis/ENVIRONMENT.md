# Software environment

Two environments were used. Exact package lists are in `requirements-katana.txt` and `requirements-local.txt`.

**GPU work: feature extraction, fine-tuning and small CNNs.**
- Ran on UNSW Katana: Python 3.11.3, torch 2.5.1+cu121, torchvision 0.20.1, numpy 1.26.4, scikit-learn 1.9.1, scipy 1.17.1.
- The full-correction permutations (`perm_full.py`) and the recipe permutations (`recipe_nonlinear_perm.py`) also ran there.

**Everything else: models on frozen features, statistics and figures.**
- Ran locally: Python 3.14.6, numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, matplotlib 3.11.0, torch 2.13.0 (CPU).

**DINOv2.**
- Code: loaded with `torch.hub.load(<local copy of facebookresearch/dinov2, main branch>, "dinov2_vitb14", source="local")`. The local copy has no git metadata; its `hubconf.py` md5 is `aea9b72b3b96c4dae212216512913df7`.
- Weights: `dinov2_vitb14_pretrain.pth`, sha256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`, the official file from dl.fbaipublicfiles.com.

**Other checkpoints.**
- torchvision ImageNet weights: `resnet18-f37072fd.pth` and `resnet50-11ad3fa6.pth`.
- MicroNet ResNet-50 encoders from NASA pretrained-microscopy-models: v1.1 for micrographs from scratch, and v1.0 for ImageNet followed by micrographs.

**Determinism.** All cross-validation splits use `RepeatedKFold(random_state=0)`, and permutations use `numpy.random.default_rng(0)`. The ridge and PCA models are deterministic. The locked-set scorers (`dev_eval.py --final`, `screen_locked.py`) refuse to run a second time.
