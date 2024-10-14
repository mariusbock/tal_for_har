# Temporal Action Localization for Inertial-based Human Activity Recognition

## Abstract
As of today, state-of-the-art activity recognition from wearable sensors relies on algorithms being trained to classify fixed windows of data. In contrast, video-based Human Activity Recognition, known as Temporal Action Localization (TAL), has followed a segment-based prediction approach, localizing activity segments in a timeline of arbitrary length. This paper is the first to systematically demonstrate the applicability of state-of-the-art TAL models for both offline and near-online Human Activity Recognition (HAR) using raw inertial data as well as pre-extracted latent features as input.  Offline prediction results show that TAL models are able to outperform popular inertial models on a multitude of HAR benchmark datasets, with improvements reaching as much as 26\% in F1-score. We show that by analyzing timelines as a whole, TAL models can produce more coherent segments and achieve higher NULL-class accuracy across all datasets. We demonstrate that TAL is less suited for the immediate classification of small-sized windows of data, yet offers an interesting perspective on inertial-based HAR -- alleviating the need for fixed-size windows and enabling algorithms to recognize activities of arbitrary length. With design choices and training concepts yet to be explored, we argue that TAL architectures could be of significant value to the inertial-based HAR community.

## Supplementary Material
Additional results and figures can be found in the `supplementary_material.pdf`.

## Installation
Please follow instructions mentioned in the [INSTALL.md](/INSTALL.md) file.

## Download
The datasets can be downloaded [here](https://uni-siegen.sciebo.de/s/BNuj9LWBaMs5tZs). The dataset download does not contain the files necessary to perform the chunked experiments. In order to create these files, please run the `data_creation.py`script once for each dataset setting the parameters as follows:

- Opportunity: `create_annotations=True`, `chunked=True`, `chunk_size=[1, 5, 30, 60]`, `window_size=30`, `window_overlap=50`
- All other datasets: `create_annotations=True`, `chunked=True`, `chunk_size=[1, 5, 30, 60]`, `window_size=50`, `window_overlap=50`

## Reproduce Experiments
Once having installed requirements, one can rerun experiments by running the `main.py` script:

````
python main.py --config ./configs/baseline/actionformer/wear_loso.yaml --seed 1
````

Each config file represents one type of experiment. Each experiment was run three times using three different random seeds (i.e. `1, 2, 3`). To rerun the experiments without changing anything about the config files, please place the complete dataset download into a folder called `data` in the main directory of the repository. Note that in order to run the two-stage training approaches (indicated by `_lstm`) the feature extraction experiments (found within the folder `feature_extraction`) need to be run first.

## Postprocessing
Please follow instructions mentioned in the [README.md](/postprocessing/README.md) file in the postprocessing subfolder. Outputted plots of the DETAD analysis performed on each algorithm and dataset can be found in the `detad_analysis` folder.

## Logging using Neptune.ai
In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project` and `api_token` information in your local deployment (see lines `29-30` in `main.py`)

## Contact
Marius Bock (marius.bock@uni-siegen.de)

## Cite as
Coming soon
