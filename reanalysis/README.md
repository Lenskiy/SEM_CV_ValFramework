# Re-analysis: from deposition to response (ZnO acetone sensors)

This folder holds the code and derived results behind the manuscript *"From deposition to response: explainable vision models and a resistance screen diagnose failing ZnO gas sensors"* (in preparation).

It re-analyses the sensors studied by the notebook in the repository root. The main difference is the evaluation:
- Every model here is tested on whole sensors it has never seen.
- 15 sensors were locked away before any modelling and scored once.
- The model choices were fixed in writing in advance.

The analysis then follows the chain recipe → microstructure (500× SEM) → baseline resistance → response.

## Layout

    code/reanalysis/   all scripts (run from this folder); README_PAPER.md maps every
                       figure and number in the manuscript to its script and output file;
                       ENVIRONMENT.md and requirements-*.txt pin the software versions
    code/reanalysis/hpc/  job scripts for the UNSW Katana cluster (PBS)
    data/derived/      derived per-sensor tables and results (CSV/JSON)
    data/dev_eval_500x/  development results, the pre-registered configuration, the selection
                       rule and its outcome, permutation nulls
    data/xai_500x/     tile-level explanations, morphology types, t-SNE

## Not included

- **Raw gas-sensing workbooks.** Published at USyd eScholarship, hdl 2123/35148. Put them in `data/new_sensor_responses/`.
- **SEM images.** Deposit pending. Point `ZNO_SEM500` to the 500× folder (one folder per sensor), or place it at `data/SEM_500x/`.
- **Image features.** These are regenerated from the images with `extract_features.py`.

## Locked test set

The 15 locked sensors (`data/derived/locked_test_sensors.txt`) were scored exactly twice. Both scorers refuse to run again:
- the image model: `dev_eval.py --final`, result in `data/derived/final_test_log.csv`;
- the resistance screen: `screen_locked.py`, under the hashed rule in `data/derived/screen_locked_rule.txt`.

