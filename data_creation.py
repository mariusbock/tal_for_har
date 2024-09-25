# ------------------------------------------------------------------------
# Data creation utilities used for creating the datasets
# ------------------------------------------------------------------------
# Adaption by: Anonyimized
# E-Mail: anonymized
# ------------------------------------------------------------------------
import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from utils.data_utils import apply_sliding_window
dataset = "hangtime"
types = ['loso']
create_annotations = True
chunked = True
chunk_size = [1]
window_size = 50
window_overlap = 50

if dataset == "rwhar":
    fps = 50
    sampling_rate = 50
    nb_sbjs = 15
    label_dict = {
        'null': 0,
        'climbingdown': 1,
        'climbingup': 2,
        'jumping': 3,
        'lying': 4,
        'running': 5,
        'sitting': 6,
        'standing': 7,
        'walking': 8
    }
    label_dict_no_null = {
        'climbingdown': 0,
        'climbingup': 1,
        'jumping': 2,
        'lying': 3,
        'running': 4,
        'sitting': 5,
        'standing': 6,
        'walking': 7
    }
elif dataset == "wetlab":
    fps = 50
    sampling_rate = 50
    nb_sbjs = 22
    label_dict = {
        'null': 0,
        'cutting': 1,
        'inverting': 2,
        'peeling': 3,
        'pestling': 4,
        'pipetting': 5,
        'pouring': 6,
        'stirring': 7,
        'transfer': 8
    }
    label_dict_no_null = {
        'cutting': 0,
        'inverting': 1,
        'peeling': 2,
        'pestling': 3,
        'pipetting': 4,
        'pouring': 5,
        'stirring': 6,
        'transfer': 7
    }   
elif dataset == "opportunity":
    fps = 30
    sampling_rate = 30
    nb_sbjs = 4
    label_dict = {
        'null': 0,
        'open_door_1': 1,
        'open_door_2': 2,
        'close_door_1': 3,
        'close_door_2': 4,
        'open_fridge': 5,
        'close_fridge': 6,
        'open_dishwasher': 7,
        'close_dishwasher': 8,
        'open_drawer_1': 9,
        'close_drawer_1': 10,
        'open_drawer_2': 11,
        'close_drawer_2': 12,
        'open_drawer_3': 13,
        'close_drawer_3': 14,
        'clean_table': 15,
        'drink_from_cup': 16,
        'toggle_switch': 17
    }
    label_dict_no_null = {
        'open_door_1': 0,
        'open_door_2': 1,
        'close_door_1': 2,
        'close_door_2': 3,
        'open_fridge': 4,
        'close_fridge': 5,
        'open_dishwasher': 6,
        'close_dishwasher': 7,
        'open_drawer_1': 8,
        'close_drawer_1': 9,
        'open_drawer_2': 10,
        'close_drawer_2': 11,
        'open_drawer_3': 12,
        'close_drawer_3': 13,
        'clean_table': 14,
        'drink_from_cup': 15,
        'toggle_switch': 16
    }
elif dataset == "sbhar":
    fps = 50
    sampling_rate = 50
    nb_sbjs = 30
    label_dict = {
        'null': 0,
        'walking': 1,
        'walking_upstairs': 2,
        'walking_downstairs': 3,
        'sitting': 4,
        'standing': 5,
        'lying': 6,
        'stand-to-sit': 7,
        'sit-to-stand': 8,
        'sit-to-lie': 9,
        'lie-to-sit': 10,
        'stand-to-lie': 11,
        'lie-to-stand': 12,
    }
    label_dict_no_null = {
        'walking': 0,
        'walking_upstairs': 1,
        'walking_downstairs': 2,
        'sitting': 3,
        'standing': 4,
        'lying': 5,
        'stand-to-sit': 6,
        'sit-to-stand': 7,
        'sit-to-lie': 8,
        'lie-to-sit': 9,
        'stand-to-lie': 10,
        'lie-to-stand': 11,
    }
elif dataset == "hangtime":
    fps = 50
    sampling_rate = 50
    nb_sbjs = 24
    label_dict = {
        'null': 0,
        'dribbling': 1,
        'shot': 2,
        'pass': 3,
        'rebound': 4,
        'layup': 5,
    }
    label_dict_no_null = {
        'dribbling': 0,
        'shot': 1,
        'pass': 2,
        'rebound': 3,
        'layup': 4,
    }
elif dataset == 'wear':
    fps = 50
    sampling_rate = 50
    nb_sbjs = 18
    label_dict = {
        'null': 0,
        'jogging': 1,
        'jogging (rotating arms)': 2,
        'jogging (skipping)': 3,
        'jogging (sidesteps)': 4,
        'jogging (butt-kicks)': 5,
        'stretching (triceps)': 6,
        'stretching (lunging)': 7,
        'stretching (shoulders)': 8,
        'stretching (hamstrings)': 9,
        'stretching (lumbar rotation)': 10,
        'push-ups': 11,
        'push-ups (complex)': 12,
        'sit-ups': 13,
        'sit-ups (complex)': 14,
        'burpees': 15,
        'lunges': 16,
        'lunges (complex)': 17,
        'bench-dips': 18
    }
    label_dict_no_null = {
        'jogging': 0,
        'jogging (rotating arms)': 1,
        'jogging (skipping)': 2,
        'jogging (sidesteps)': 3,
        'jogging (butt-kicks)': 4,
        'stretching (triceps)': 5,
        'stretching (lunging)': 6,
        'stretching (shoulders)': 7,
        'stretching (hamstrings)': 8,
        'stretching (lumbar rotation)': 9,
        'push-ups': 10,
        'push-ups (complex)': 11,
        'sit-ups': 12,
        'sit-ups (complex)': 13,
        'burpees': 14,
        'lunges': 15,
        'lunges (complex)': 16,
        'bench-dips': 17
    }

def convert_labels_to_annotation_json(labels, sr, duration_seconds):
    thumos_annotations = []
    curr_start_i = 0
    curr_end_i = 0
    curr_label = labels.iloc[0]
    num_instances = 0
    
    for i, l in enumerate(labels):
        if curr_label != l or curr_end_i == len(labels) - 1:
            act_start = curr_start_i / sr
            act_end = curr_end_i / sr
            act_label = curr_label
            # create annotation
            if act_label != "null" and not pd.isnull(act_label):
                length = (act_end - act_start)
                coverage = (act_end - act_start) / duration_seconds
                num_instances += 1
                thumos_anno = {
                    "all-segments": None,
                    "agreement": None,
                    "label": act_label,
                    "length": length,
                    "coverage": coverage,
                    "context-distance": None,
                    "segment": [
                        act_start,
                        act_end
                    ],
                    "segment (frames)": [
                        act_start * fps,
                        act_end * fps
                    ],
                    "label_id": label_dict[act_label] - 1,
                    "context-size": None
                }
                thumos_annotations.append(thumos_anno)  
            curr_label = l
            curr_start_i = i + 1
            curr_end_i = i + 1
        else:
            curr_end_i += 1
        thumos_annotations = [dict(item, **{'num-instances': num_instances}) for item in thumos_annotations]
    return thumos_annotations

if not chunked:
    chunk_size = [0]
for chunk_s in chunk_size:    
    for i in range(nb_sbjs):
        thumos_annotations = {"version": [], "taxonomy": [], "database": {}, "label_dict": label_dict_no_null}
        chunked_annotations = {"version": [], "taxonomy": [], "database": {}, "label_dict": label_dict_no_null}
        print('PROCESSING: sbj_{}'.format(int(i)))   
        raw_sbj = pd.read_csv(os.path.join('./data/' + dataset + '/raw/inertial', 'sbj_' + str(int(i)) + '.csv'), index_col=None, low_memory=False)
        columns = raw_sbj.columns
        sens_sbj = raw_sbj.replace({"label": label_dict}).fillna(0).to_numpy()
        if chunked:
            for k in range(0, len(sens_sbj), chunk_s * sampling_rate):
                if k + 2 * (chunk_s * sampling_rate) > len(sens_sbj):
                    batch_start  = k
                    batch_end = len(sens_sbj)
                else:
                    batch_start = k
                    batch_end = k + (chunk_s * sampling_rate)
                chunked_sbj = raw_sbj.iloc[batch_start:batch_end, :]
                chunked_path = os.path.join('./data/' + dataset + '/raw/', f'inertial')
                if not os.path.exists(chunked_path):
                    os.makedirs(chunked_path)
                chunked_sbj.to_csv(os.path.join(chunked_path, 'sbj_' + str(int(i)) + '_chunk_' + str(batch_start) + '_' + str(batch_end) + '.csv'), index=False, header=columns)
                chunked_sbj = chunked_sbj.replace({"label": label_dict}).fillna(0).to_numpy()
                _, c_win_sbj, _ = apply_sliding_window(chunked_sbj, window_size, window_overlap)
                c_flipped_sbj = np.transpose(c_win_sbj[:, :, 1:], (0,2,1))
                c_flat_flipped_sbj = c_flipped_sbj.reshape(c_flipped_sbj.shape[0], -1)
                c_processed_folder = os.path.join('./data/' + dataset + '/processed/inertial_features', '{}_samples_{}_overlap'.format(window_size, window_overlap))
                if not os.path.exists(c_processed_folder):
                    os.makedirs(c_processed_folder)
                np.save(os.path.join(c_processed_folder,'sbj_' + str(int(i)) + '_chunk_' + str(batch_start) + '_' + str(batch_end) + '.npy'), c_flat_flipped_sbj)
                if k + 2 * (chunk_s * sampling_rate) > len(sens_sbj):
                    break

        _, win_sbj, _ = apply_sliding_window(sens_sbj, window_size, window_overlap)
        flipped_sbj = np.transpose(win_sbj[:, :, 1:], (0,2,1))
        flat_flipped_sbj = flipped_sbj.reshape(flipped_sbj.shape[0], -1)
        processed_folder = os.path.join('./data/' + dataset + '/processed/inertial_features', '{}_samples_{}_overlap'.format(window_size, window_overlap))
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        np.save(os.path.join(processed_folder,'sbj_' + str(int(i)) + '.npy'), flat_flipped_sbj)
        
        if create_annotations:
            for j in range(nb_sbjs):
                sbj_data = pd.read_csv(os.path.join('data', dataset, 'raw/inertial', 'sbj_' + str(int(j)) + '.csv'), index_col=None, low_memory=False)
                # create video annotations
                duration_seconds = len(sbj_data) / sampling_rate
                sbj_thumos_annos = convert_labels_to_annotation_json(sbj_data.iloc[:, -1], sampling_rate, duration_seconds)
                if i == j:
                    train_test = "Validation"
                    if chunked:
                        for k in range(0, len(sbj_data), chunk_s * sampling_rate):
                            if k + 2 * (chunk_s * sampling_rate) > len(sens_sbj):
                                batch_start  = k
                                batch_end = len(sbj_data)
                            else:
                                batch_start = k
                                batch_end = k + (chunk_s * sampling_rate)
                            chunked_annos = convert_labels_to_annotation_json(sbj_data.iloc[batch_start:batch_end, -1], sampling_rate, duration_seconds)
                            chunked_annotations['database']['sbj_' + str(int(j)) + '_chunk_' + str(batch_start) + '_' + str(batch_end)] = {
                                "subset": train_test,
                                "duration": float(chunk_s),
                                "fps": fps,
                                "annotations": chunked_annos,
                                }
                            if k + 2 * (chunk_s * sampling_rate) > len(sens_sbj):
                                break 
                else:
                    train_test = "Training"
                    chunked_annos = sbj_thumos_annos
                    chunked_annotations['database']['sbj_' + str(int(j))] = {
                    "subset": train_test,
                    "duration": duration_seconds,
                    "fps": fps,
                    "annotations": sbj_thumos_annos,
                    } 
                thumos_annotations['database']['sbj_' + str(int(j))] = {
                    "subset": train_test,
                    "duration": duration_seconds,
                    "fps": fps,
                    "annotations": sbj_thumos_annos,
                    } 
                
            with open('data/' + dataset +'/annotations/' + 'loso_' + 'sbj_' + str(int(i)) +  '.json', 'w') as file1:
                file1.write(json.dumps(thumos_annotations, indent = 4))
            if chunked:
                chunked_annotations_folder = 'data/' + dataset + '/annotations/chunked/' + str(chunk_s)
                if not os.path.exists(chunked_annotations_folder):
                    os.makedirs(chunked_annotations_folder)
                with open(os.path.join(chunked_annotations_folder, 'loso_sbj_' + str(int(i)) + '.json'), 'w') as file2:
                    file2.write(json.dumps(chunked_annotations, indent=4))
