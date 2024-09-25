# Postprocessing of experiments

## Inertial-based Architecture Experiments
Run the `majority_fil.py` script by changing the `path_to_preds` path-variable pointing towards a folder containing the logged experiments as separate folders, i.e.: 
- If you ran three experiments with varying different seeds place all three folders in a directory.
- Name each folder following the name structure `seed_X` where `X` is the employed seed of each experiment.
- Define the `seeds` and `majority_filters` you want to test.

## Camera-based Architecture Experiments
Run the `score_thres.py` script by changing the `path_to_preds` path-variable pointing towards a folder containing the logged experiments as separate folders, i.e.: 
- If you ran three experiments with varying different seeds place all three folders in a directory.
- Name each folder following the name structure `seed_X` where `X` is the employed seed of each experiment.
- Define the `seeds` and `score_thresholds` you want to test.

## DETAD analysis
Run the `run_detad_analysis.py` script by changing the `path_to_preds`, `output_folder` and `gt_file` path-variables pointing towards your predictions and dataset directories. The scripts were taken from the [original paper](https://github.com/HumamAlwassel/DETAD) and marginally adopted to ensure readability. The scripts will output plots as described in the original repository in the defined output folder.
