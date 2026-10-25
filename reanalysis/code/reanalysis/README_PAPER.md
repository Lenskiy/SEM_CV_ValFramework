# From deposition to response: code map

This README maps each result in the paper to the script that produces it and the file it writes. Run every script from `code/reanalysis/`. Paths are relative to the repository root.

All analyses use the 56 development sensors only, unless a script says otherwise. The 15 locked sensors were scored exactly twice, and both scorers refuse to run again:
- the image model: `dev_eval.py --final` → `data/derived/final_test_log.csv`;
- the resistance screen: `screen_locked.py` → `data/derived/screen_locked_log.json`, under the hashed rule in `data/derived/screen_locked_rule.txt`.

The environment is described in `ENVIRONMENT.md`.

## Inputs
| Data | Produced by |
|---|---|
| Labels, from the gas-test workbooks (`data/new_sensor_responses/*.xlsx`) | `labels_v2.py` → `data/derived/labels_v2.csv` |
| Baseline resistance | `robustness.py` (E3) → `data/derived/r_air.csv` |
| Image features: 500× images → tiles → frozen backbones | `extract_features.py` → `data/features500x/*.npz`, `images.csv` |
| Split: development vs locked, fixed before any image model | `data/derived/locked_test_sensors.txt` |

## Results, figures and tables
| Paper item | Script | Output |
|---|---|---|
| Fig. 1; repeatability (ICC) | `labels_v2.py`, `figures.py:fig_recipes` | `labels_v2.csv` |
| Resistance vs response; Fig. 2 | `robustness.py` (E3), `figures.py:fig_rair` | `robustness.json` |
| Resistance screen: development; non-responder thresholds | `screen.py`, `robustness_chain.py` (R5) | `screen.json`, `robustness_chain.json` |
| Resistance screen: locked check (once) | `screen_locked.py` | `screen_locked_log.json` |
| Sign of the resistance change; signed label | `sign_label.py` | `sign_label.json`, `sign_label.csv` |
| Oxygen limit, zinc thickness h | `regime.py` | `regime.json`, `regime_sensors.csv` |
| Chain: recipe / images → resistance → response; mediation; Fig. 3 | `chain.py`, `figures.py:fig_chain` | `chain.json`, `chain_oof.csv` |
| Gap obscuration | `bridging.py`, `bridging_stats.py` | `bridging.csv`, `bridging_stats.json` |
| Image → resistance (all backbones) | `r_air_predict.py`, `r_air_backbones.py` | `r_air_predict.json`, `r_air_backbones.csv` |
| Robustness: bootstrap interval, August images, unseen levels, gap settings | `robustness_chain.py` | `robustness_chain.json` |
| Nonlinear recipe baselines, their permutation null, unseen levels | `recipe_nonlinear.py`, `recipe_nonlinear_perm.py`, `recipe_lolo.py` | `recipe_nonlinear*.json`, `recipe_lolo.json` |
| Imaging-session check | `session_check.py` | `session_check.json` |
| Excluded sensor put back (57 sensors) | `outlier57.py` | `outlier57.json` |
| Response models: development; pre-registered config | `dev_eval.py` | `data/dev_eval_500x/*` |
| Full correction over 67 configurations | `perm_full.py` (Katana: `hpc/perm_full.pbs`) | `data/dev_eval_500x/perm_full_response.csv` |
| Fine-tuning and the selection rule | `finetune.py`; rule in `selection_rule_500x.txt` | `selection_outcome_500x.txt` |
| Small CNNs and two-head networks; Table S1 | `train_small.py`, `compare_small.py`, `compare_dual.py`, `vitb_classes.py` | `compare_small.csv`, `compare_dual.csv`, `vitb_classes.json` |
| Fig. 4 | `figures.py:fig_models` | `paper_v2/figures/fig_models.pdf` |
| Replicates; Fig. 5 | `replicate_eval.py`, `figures.py:fig_replicates` | (values printed by `replicate_eval.py`) |
| Tile explanation, response model; morphology types | `xai_tiles.py` | `data/xai_500x/tile_scores.csv`, `type_summary.csv` |
| Tile explanation, resistance model; tile removal; Figs 6–7 | `xai_rair.py`, `robustness.py` (E4), `figures.py:fig_tilemaps`, `fig_types` | `data/xai_500x/xai_rair.json`, `tile_scores_rair.csv` |
| Kinetics gate | `kinetics.py`, `kinetics_sensitivity.py` | `kinetics_gate.json`, `kinetics_sensitivity.csv` |
| t-SNE and neighbours; Supplementary Fig. S1 | `tsne_check.py`, `figures.py:fig_tsne` | `data/xai_500x/tsne_*` |
| Sensor-level PCA; Supplementary Fig. S2 | `figures.py:fig_pca` | `paper_v2/figures/fig_pca.pdf` |
| Heating curves; Supplementary Fig. S3 | `heating_rt.py`, `figures.py:fig_heating` | `heating_rt.json`, `heating_rt.csv` |
| Per-sensor table; Supplementary Table S2 | `sensor_table.py` | `sensor_table.csv`, `paper_v2/table_s2.tex` |

To regenerate every figure, run `python figures.py`.
