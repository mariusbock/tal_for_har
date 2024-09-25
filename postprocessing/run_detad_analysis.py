# ------------------------------------------------------------------------
# Main script to run DETAD analysis based on publication by Alwassel et al.
# https://github.com/HumamAlwassel/DETAD
# ------------------------------------------------------------------------
# Adaption by: anonymized
# E-Mail: anonymized
# ------------------------------------------------------------------------
import glob
import itertools
import pandas as pd
import numpy as np
import os, json

from detad_analysis.false_negative_analysis import false_negative_analysis
from detad_analysis.false_positive_analysis import false_positive_analysis
from detad_analysis.sensitivity_analysis import sensitivity_analysis
from utils import convert_samples_to_segments, majorityVoting

# hangtime error bei orig_deepconvlstm
datasets = ['hangtime', 'rwhar', 'wear', 'wetlab', 'opportunity', 'sbhar']
#algorithms = ['actionformer', 'temporalmaxer', 'tridet', 'dim_shallow_deepconvlstm', 'orig_aandd', 'tinyhar']
algorithms = ['orig_deepconvlstm']

for dataset, algorithm in list(itertools.product(datasets, algorithms)):
    path_to_preds = 'experiments/{}/{}'.format(dataset, algorithm)
    seed = 1
    output_folder = 'detad_plots/{}/{}'.format(dataset, algorithm)
    gt_file = 'data/{}/annotations/eval_all.json'.format(dataset)

    if dataset == "rwhar":
        score_thres = 0.0
        majority_filter = 2001
        sampling_rate = 50
        label_dict = {
            0: 'climbingdown',
            1: 'climbingup',
            2: 'jumping',
            3: 'lying',
            4: 'running',
            5: 'sitting',
            6: 'standing',
            7: 'walking'
        }
    elif dataset == "wetlab":
        score_thres = 0.15
        majority_filter = 1001
        sampling_rate = 50
        label_dict = {
            0: 'cutting',
            1: 'inverting',
            2: 'peeling',
            3: 'pestling',
            4: 'pipetting',
            5: 'pouring',
            6: 'stirring',
            7: 'transfer'
        }   
    elif dataset == "opportunity":
        score_thres = 0.2
        majority_filter = 76
        sampling_rate = 30
        label_dict = {
            0: 'open_door_1',
            1: 'open_door_2',
            2: 'close_door_1',
            3: 'close_door_2',
            4: 'open_fridge',
            5: 'close_fridge',
            6: 'open_dishwasher',
            7: 'close_dishwasher',
            8: 'open_drawer_1',
            9: 'close_drawer_1',
            10: 'open_drawer_2',
            11: 'close_drawer_2',
            12: 'open_drawer_3',
            13: 'close_drawer_3',
            14: 'clean_table',
            15: 'drink_from_cup',
            16: 'toggle_switch'
        }
    elif dataset == "sbhar":
        score_thres = 0.3
        majority_filter = 251
        sampling_rate = 50
        label_dict = {
            0: 'walking',
            1: 'walking_upstairs',
            2: 'walking_downstairs',
            3: 'sitting',
            4: 'standing',
            5: 'lying',
            6: 'stand-to-sit',
            7: 'sit-to-stand',
            8: 'sit-to-lie',
            9: 'lie-to-sit',
            10: 'stand-to-lie',
            11: 'lie-to-stand',
        }
    elif dataset == "hangtime":
        score_thres = 0.15
        majority_filter = 751
        sampling_rate = 50
        label_dict = {
            0: 'dribbling',
            1: 'shot',
            2: 'pass',
            3: 'rebound',
            4: 'layup',
        }
    elif dataset == 'wear':
        score_thres = 0.2
        majority_filter = 251
        sampling_rate = 50
        label_dict = {
            0: 'jogging',
            1: 'jogging (rotating arms)',
            2: 'jogging (skipping)',
            3: 'jogging (sidesteps)',
            4:'jogging (butt-kicks)',
            5: 'stretching (triceps)',
            6: 'stretching (lunging)',
            7: 'stretching (shoulders)',
            8: 'stretching (hamstrings)',
            9: 'stretching (lumbar rotation)',
            10: 'push-ups',
            11: 'push-ups (complex)',
            12: 'sit-ups',
            13: 'sit-ups (complex)',
            14: 'burpees',
            15: 'lunges',
            16: 'lunges (complex)',
            17: 'bench-dips'
        }

    print(algorithm, dataset)
    predictions = {"version": [], "external_data": [], "results": {}}
    if algorithm in ['actionformer', 'temporalmaxer', 'tridet']:
        pred_files = sorted(glob.glob(os.path.join(path_to_preds, 'seed_{}'.format(seed), 'unprocessed_results', 'v_seg_*.csv')))
    else:
        pred_files = sorted(glob.glob(os.path.join(path_to_preds, 'seed_{}'.format(seed), 'unprocessed_results', 'v_preds_*.npy')))
    for file in pred_files:
        if algorithm in ['actionformer', 'temporalmaxer', 'tridet']:
            v_seg = pd.read_csv(file)
            v_seg = v_seg[v_seg.score > score_thres]
        else:
            v_orig_preds = np.load(file)
            v_fil_preds = [majorityVoting(i, v_orig_preds.astype(int), majority_filter) for i in range(len(v_orig_preds))]
            sbj_num = file.split('/')[-1].split('.')[0].split('_')[-1]
            v_seg = convert_samples_to_segments([sbj_num], v_fil_preds, sampling_rate)
            v_seg = pd.DataFrame(v_seg)
        v_seg = v_seg.rename(columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})
        video_id = v_seg['video-id'][0]
        predictions['results'][video_id] = json.loads(v_seg[['t-start', 't-end', 'label', 'score']].replace({"label": label_dict}).to_json(orient='records'))


    false_negative_analysis(gt_file, 'Validation', predictions, output_folder, True)
    false_positive_analysis(gt_file, 'Validation', predictions, output_folder, True)
    sensitivity_analysis(gt_file, 'Validation', predictions, output_folder, True)

                

