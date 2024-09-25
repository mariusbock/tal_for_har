# ------------------------------------------------------------------------
# Data operation utilities
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------

import numpy as np
import pandas as pd


def sliding_window_samples(data, win_len, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param length_in_seconds: int, default: 1
        Window length as seconds
    :param sampling_rate: int, default: 50
        Sampling rate in hertz as integer value
    :param overlap_ratio: int, default: None
        Overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0

    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        else:
            float_prec = False
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True

    return np.array(windows), np.array(indices)

def apply_sliding_window(data, window_size, window_overlap):
    output_x = None
    output_y = None
    output_sbj = []
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)

        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)

    output_y = [[i[-1]] for i in output_y]
    return output_sbj, output_x, np.array(output_y).flatten()

def unwindow_inertial_data(orig, ids, preds, win_size, win_overlap):
    unseg_preds = []

    if not ((win_overlap / 100) * win_size).is_integer():
        float_prec = True
    else:
        float_prec = False

    for sbj in np.unique(orig[:, 0]):
        sbj_data = orig[orig[:, 0] == sbj]
        sbj_preds = preds[ids==sbj]
        sbj_unseg_preds = []
        changing_bool = True
        for i, pred in enumerate(sbj_preds):
            if (float_prec == True) and (changing_bool == True):
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)) + 1)))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (win_overlap * 0.01)) + 1)))
                changing_bool = False
            else:
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)))))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * int(win_size * (win_overlap * 0.01))))
                changing_bool = True
        sbj_unseg_preds = np.concatenate((sbj_unseg_preds, np.full(len(sbj_data) - len(sbj_unseg_preds), sbj_preds[-1])))
        unseg_preds = np.concatenate((unseg_preds, sbj_unseg_preds))
    assert len(unseg_preds) == len(orig)    
    return unseg_preds, orig[:, -1]

def convert_samples_to_segments(ids, labels, sampling_rate):
    f_video_ids, f_labels, f_t_start, f_t_end, f_score = [], np.array([]), np.array([]), np.array([]), np.array([])

    for id in np.unique(ids):
        sbj_labels = labels[(ids == id)]
        curr_start_i = 0
        curr_end_i = 0
        curr_label = sbj_labels[0]
        for i, l in enumerate(sbj_labels):
            if curr_label != l:
                act_start = curr_start_i / sampling_rate
                act_end = curr_end_i / sampling_rate
                act_label = curr_label - 1
                if curr_label != 0:
                    # create annotation
                    f_video_ids.append('sbj_' + str(int(id)))
                    f_labels = np.append(f_labels, act_label)
                    f_t_start = np.append(f_t_start, act_start)
                    f_t_end = np.append(f_t_end, act_end)
                    f_score = np.append(f_score, 1)
                curr_label = l
                curr_start_i = i + 1
                curr_end_i = i + 1    
            else:
                curr_end_i += 1        
    return {
        'video-id': f_video_ids,
        'label': f_labels,
        't-start': f_t_start,
        't-end': f_t_end,
        'score': f_score
    }

def convert_segments_to_samples(segments, sens, sampling_rate, threshold_type='score', threshold=0.0):
    if isinstance(segments, pd.DataFrame):
        segments_df = segments
    else:
        try:
            segments_df = pd.DataFrame({
                'video_id' : segments['video-id'],
                't_start' : segments['t-start'].tolist(),
                't_end': segments['t-end'].tolist(),
                'label': segments['label'].tolist(),
                'score': segments['score'].tolist()
                })
        except:
            segments_df = pd.DataFrame(columns=['video_id', 't_start', 't_end', 'label', 'score'])
    
    if threshold_type == 'score':
        fil_sbj_segments = segments.loc[segments_df.score > threshold]
        
    if len(fil_sbj_segments) == 0:
        fil_sbj_segments = segments
             
    gt = sens[:, -1]
    preds = np.empty(len(sens))
    preds[:] = np.nan
    scores = np.zeros(len(sens))
            
    for _, seg in fil_sbj_segments.iterrows():
        preds[int(np.floor(seg['t_start'] * sampling_rate)):int(np.ceil(seg['t_end'] * sampling_rate))] = seg['label'] + 1
        scores[int(np.floor(seg['t_start'] * sampling_rate)):int(np.ceil(seg['t_end'] * sampling_rate))] = seg['score']
            
    preds = np.nan_to_num(preds)
    return preds, gt, scores
