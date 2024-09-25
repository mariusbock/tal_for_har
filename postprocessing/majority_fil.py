# ------------------------------------------------------------------------
# Postprocessing script used for majority filtering inertial predictions
# ------------------------------------------------------------------------
# Adaption by: anonymized
# E-Mail: anonymized
# ------------------------------------------------------------------------
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

from utils import majorityVoting, ANETdetection, convert_samples_to_segments, compute_misalignment_measures
from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# postprocessing parameters
dataset = 'wear'
path_to_preds = ['experiments/baseline/{}/tinyhar'.format(dataset)]
seeds = [1, 2, 3]
#majority_filters = [1, 251, 501, 751, 1001, 1251]
majority_filters = [751]

if dataset == 'opportunity':
    # all == 76 
    #majority_filters = [1, 76, 151, 301, 451, 501]
    majority_filters = [76]
    num_classes = 18
    sampling_rate = 30
    input_dim = 113
    json_files = [
    'data/opportunity/annotations/loso_sbj_0.json',
    'data/opportunity/annotations/loso_sbj_1.json',
    'data/opportunity/annotations/loso_sbj_2.json',
    'data/opportunity/annotations/loso_sbj_3.json',
    ]
elif dataset == 'wetlab':
    # all == 1001 
    num_classes = 9
    sampling_rate = 50
    input_dim = 3
    json_files = [
    'data/wetlab/annotations/loso_sbj_0.json',
    'data/wetlab/annotations/loso_sbj_1.json',
    'data/wetlab/annotations/loso_sbj_2.json',
    'data/wetlab/annotations/loso_sbj_3.json',
    'data/wetlab/annotations/loso_sbj_4.json',
    'data/wetlab/annotations/loso_sbj_5.json',
    'data/wetlab/annotations/loso_sbj_6.json',
    'data/wetlab/annotations/loso_sbj_7.json',
    'data/wetlab/annotations/loso_sbj_8.json',
    'data/wetlab/annotations/loso_sbj_9.json',
    'data/wetlab/annotations/loso_sbj_10.json',
    'data/wetlab/annotations/loso_sbj_11.json',
    'data/wetlab/annotations/loso_sbj_12.json',
    'data/wetlab/annotations/loso_sbj_13.json',
    'data/wetlab/annotations/loso_sbj_14.json',
    'data/wetlab/annotations/loso_sbj_15.json',
    'data/wetlab/annotations/loso_sbj_16.json',
    'data/wetlab/annotations/loso_sbj_17.json',
    'data/wetlab/annotations/loso_sbj_18.json',
    'data/wetlab/annotations/loso_sbj_19.json',
    'data/wetlab/annotations/loso_sbj_20.json',
    'data/wetlab/annotations/loso_sbj_21.json'
    ]
elif dataset == 'sbhar':
    # all == 251 
    num_classes = 13
    sampling_rate = 50
    input_dim = 3
    json_files = [
    'data/sbhar/annotations/loso_sbj_0.json',
    'data/sbhar/annotations/loso_sbj_1.json',
    'data/sbhar/annotations/loso_sbj_2.json',
    'data/sbhar/annotations/loso_sbj_3.json',
    'data/sbhar/annotations/loso_sbj_4.json',
    'data/sbhar/annotations/loso_sbj_5.json',
    'data/sbhar/annotations/loso_sbj_6.json',
    'data/sbhar/annotations/loso_sbj_7.json',
    'data/sbhar/annotations/loso_sbj_8.json',
    'data/sbhar/annotations/loso_sbj_9.json',
    'data/sbhar/annotations/loso_sbj_10.json',
    'data/sbhar/annotations/loso_sbj_11.json',
    'data/sbhar/annotations/loso_sbj_12.json',
    'data/sbhar/annotations/loso_sbj_13.json',
    'data/sbhar/annotations/loso_sbj_14.json',
    'data/sbhar/annotations/loso_sbj_15.json',
    'data/sbhar/annotations/loso_sbj_16.json',
    'data/sbhar/annotations/loso_sbj_17.json',
    'data/sbhar/annotations/loso_sbj_18.json',
    'data/sbhar/annotations/loso_sbj_19.json',
    'data/sbhar/annotations/loso_sbj_20.json',
    'data/sbhar/annotations/loso_sbj_21.json',
    'data/sbhar/annotations/loso_sbj_22.json',
    'data/sbhar/annotations/loso_sbj_23.json',
    'data/sbhar/annotations/loso_sbj_24.json',
    'data/sbhar/annotations/loso_sbj_25.json',
    'data/sbhar/annotations/loso_sbj_26.json',
    'data/sbhar/annotations/loso_sbj_27.json',
    'data/sbhar/annotations/loso_sbj_28.json',
    'data/sbhar/annotations/loso_sbj_29.json'
    ]
elif dataset == 'rwhar':
    # all == 2001 best results, but probably can be even larger
    # majority_filters = [1251, 1501, 1751, 2001]
    majority_filters = [2001]
    num_classes = 8
    sampling_rate = 50
    input_dim = 21
    json_files = [
    'data/rwhar/annotations/loso_sbj_0.json',
    'data/rwhar/annotations/loso_sbj_1.json',
    'data/rwhar/annotations/loso_sbj_2.json',
    'data/rwhar/annotations/loso_sbj_3.json',
    'data/rwhar/annotations/loso_sbj_4.json',
    'data/rwhar/annotations/loso_sbj_5.json',
    'data/rwhar/annotations/loso_sbj_6.json',
    'data/rwhar/annotations/loso_sbj_7.json',
    'data/rwhar/annotations/loso_sbj_8.json',
    'data/rwhar/annotations/loso_sbj_9.json',
    'data/rwhar/annotations/loso_sbj_10.json',
    'data/rwhar/annotations/loso_sbj_11.json',
    'data/rwhar/annotations/loso_sbj_12.json',
    'data/rwhar/annotations/loso_sbj_13.json',
    'data/rwhar/annotations/loso_sbj_14.json'
    ]
elif dataset == 'wear':
    # all == 751
    num_classes = 19
    sampling_rate = 50
    input_dim = 12
    json_files = [
    'data/wear/annotations/loso_sbj_0.json',
    'data/wear/annotations/loso_sbj_1.json',
    'data/wear/annotations/loso_sbj_2.json',
    'data/wear/annotations/loso_sbj_3.json',
    'data/wear/annotations/loso_sbj_4.json',
    'data/wear/annotations/loso_sbj_5.json',
    'data/wear/annotations/loso_sbj_6.json',
    'data/wear/annotations/loso_sbj_7.json',
    'data/wear/annotations/loso_sbj_8.json',
    'data/wear/annotations/loso_sbj_9.json',
    'data/wear/annotations/loso_sbj_10.json',
    'data/wear/annotations/loso_sbj_11.json',
    'data/wear/annotations/loso_sbj_12.json',
    'data/wear/annotations/loso_sbj_13.json',
    'data/wear/annotations/loso_sbj_14.json',
    'data/wear/annotations/loso_sbj_15.json',
    'data/wear/annotations/loso_sbj_16.json',
    'data/wear/annotations/loso_sbj_17.json'
    ]
elif dataset == 'hangtime':
    # all == 251
    num_classes = 7
    sampling_rate = 50
    input_dim = 3
    json_files = [
    'data/hangtime/annotations/loso_sbj_0.json',
    'data/hangtime/annotations/loso_sbj_1.json',
    'data/hangtime/annotations/loso_sbj_2.json',
    'data/hangtime/annotations/loso_sbj_3.json',
    'data/hangtime/annotations/loso_sbj_4.json',
    'data/hangtime/annotations/loso_sbj_5.json',
    'data/hangtime/annotations/loso_sbj_6.json',
    'data/hangtime/annotations/loso_sbj_7.json',
    'data/hangtime/annotations/loso_sbj_8.json',
    'data/hangtime/annotations/loso_sbj_9.json',
    'data/hangtime/annotations/loso_sbj_10.json',
    'data/hangtime/annotations/loso_sbj_11.json',
    'data/hangtime/annotations/loso_sbj_12.json',
    'data/hangtime/annotations/loso_sbj_13.json',
    'data/hangtime/annotations/loso_sbj_14.json',
    'data/hangtime/annotations/loso_sbj_15.json',
    'data/hangtime/annotations/loso_sbj_16.json',
    'data/hangtime/annotations/loso_sbj_17.json',
    'data/hangtime/annotations/loso_sbj_18.json',
    'data/hangtime/annotations/loso_sbj_19.json',
    'data/hangtime/annotations/loso_sbj_20.json',
    'data/hangtime/annotations/loso_sbj_21.json',
    'data/hangtime/annotations/loso_sbj_22.json',
    'data/hangtime/annotations/loso_sbj_23.json',
    ]
    
#print("Data Loading....")
for path in path_to_preds:
    for f in majority_filters:
        all_mAP = np.zeros((len(seeds), 5))
        if dataset == 'rwhar':
            all_ur = np.zeros((len(seeds), num_classes))
            all_dr = np.zeros((len(seeds), num_classes))
            all_fr = np.zeros((len(seeds), num_classes))
            all_ir = np.zeros((len(seeds), num_classes))
            all_or = np.zeros((len(seeds), num_classes))
            all_mr = np.zeros((len(seeds), num_classes))
        else:
            all_ur = np.zeros((len(seeds), num_classes - 1))
            all_dr = np.zeros((len(seeds), num_classes - 1))
            all_fr = np.zeros((len(seeds), num_classes - 1))
            all_ir = np.zeros((len(seeds), num_classes - 1))
            all_or = np.zeros((len(seeds), num_classes - 1))
            all_mr = np.zeros((len(seeds), num_classes - 1))
        all_recall = np.zeros((len(seeds), num_classes))
        all_prec = np.zeros((len(seeds), num_classes))
        all_f1 = np.zeros((len(seeds), num_classes))
        for s_pos, seed in enumerate(seeds):
            all_preds = np.array([])
            all_gt = np.array([])

            for i, j in enumerate(json_files):
                with open(j) as fi:
                    file = json.load(fi)
                    anno_file = file['database']
                    if dataset == 'rwhar':
                        labels = list(file['label_dict'])
                    else:
                        labels = ['null'] + list(file['label_dict'])
                    label_dict = dict(zip(labels, list(range(len(labels)))))
                    val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
            
                v_data = np.empty((0, input_dim + 2))
                
                for sbj in val_sbjs:
                    data = pd.read_csv(os.path.join('data/{}/raw/inertial'.format(dataset), sbj + '.csv'), index_col=False, low_memory=False).replace({"label": label_dict}).fillna(0).to_numpy()
                    v_data = np.append(v_data, data, axis=0)
                
                v_preds = np.array([])
                if 'loso' in j:
                    v_orig_preds = np.load(os.path.join(path, 'seed_' + str(seed), 'unprocessed_results/v_preds_loso_sbj_{}.npy'.format(int(i))))
                else:
                    v_orig_preds = np.load(os.path.join(path, 'seed_' + str(seed), 'unprocessed_results/v_preds_split_{}.npy'.format(int(i) + 1)))
                
                for sbj in val_sbjs:
                    sbj_pred = v_orig_preds[v_data[:, 0] == int(sbj.split("_")[-1])]
                    sbj_pred = [majorityVoting(i, sbj_pred.astype(int), f) for i in range(len(sbj_pred))]
                    v_preds = np.append(v_preds, sbj_pred)

                #print("Converting to Segments....")
                seg_data = convert_samples_to_segments(v_data[:, 0], v_preds, sampling_rate)
                det_eval = ANETdetection(j, 'validation', tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7])
                all_preds = np.concatenate((all_preds, v_preds))
                all_gt = np.concatenate((all_gt, v_data[:, -1]))

                #print("Evaluating....")
                v_mAP, _ = det_eval.evaluate(seg_data)
                if dataset == "rwhar":
                    v_ur, v_dr, v_fr, v_ir, v_or, v_mr = compute_misalignment_measures(v_data[:, -1], v_preds, range(0, num_classes), False)
                else:
                    v_ur, v_dr, v_fr, v_ir, v_or, v_mr = compute_misalignment_measures(v_data[:, -1], v_preds, range(0, num_classes))
                v_prec = precision_score(v_data[:, -1], v_preds, average=None, labels=range(0, num_classes))
                v_rec = recall_score(v_data[:, -1], v_preds, average=None, labels=range(0, num_classes))
                v_f1 = f1_score(v_data[:, -1], v_preds, average=None, labels=range(0, num_classes))
                    
                all_prec[s_pos, :] += v_prec
                all_recall[s_pos, :] += v_rec
                all_f1[s_pos, :] += v_f1
                all_ur[s_pos, :] += v_ur
                all_dr[s_pos, :] += v_dr
                all_fr[s_pos, :] += v_fr
                all_ir[s_pos, :] += v_ir
                all_or[s_pos, :] += v_or
                all_mr[s_pos, :] += v_mr
                all_mAP[s_pos, :] += v_mAP

            if seed == 1:
                comb_conf = confusion_matrix(all_gt, all_preds, normalize='true')
                comb_conf = np.around(comb_conf, 2)
                comb_conf[comb_conf == 0] = np.nan

                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                            'fontsize': 16,
                        })
                ax.set_title('Confusion Matrix')
                pred_name = path.split('/')[-2]
                _.savefig(pred_name + ".pdf")
                np.save("viz", all_preds)
                    
        print("Prediction for {} with threshold {}:".format(path_to_preds, f))
        #print("Individual mAP:")
        #print(np.around(np.mean(all_mAP, axis=0) / len(json_files), 4) * 100)

        print("Average mAP:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_mAP) / len(json_files) * 100, np.std(np.mean(all_mAP, axis=1) / len(json_files)) * 100))

        print("Average Underfill Ratio:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_ur) / len(json_files) * 100, np.std(np.mean(all_ur, axis=1) / len(json_files)) * 100))
        
        print("Average Overfill Ratio:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_or) / len(json_files) * 100, np.std(np.mean(all_or, axis=1) / len(json_files)) * 100))
        
        print("Average Deletion Ratio:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_dr) / len(json_files) * 100, np.std(np.mean(all_dr, axis=1) / len(json_files)) * 100))
        
        print("Average Insertion Ratio:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_ir) / len(json_files) * 100, np.std(np.mean(all_ir, axis=1) / len(json_files)) * 100))

        print("Average Fragmentation Ratio:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_fr) / len(json_files) * 100, np.std(np.mean(all_fr, axis=1) / len(json_files)) * 100))
        
        print("Average Merge Ratio:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_mr) / len(json_files) * 100, np.std(np.mean(all_mr, axis=1) / len(json_files)) * 100))
        
        #print("Individual Precision:")
        #print(np.around(np.mean(all_prec, axis=0) / len(json_files), 4) * 100)

        print("Average Precision:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_prec) / len(json_files) * 100, np.std(np.mean(all_prec, axis=1) / len(json_files)) * 100))

        #print("Individual Recall:")
        #print(np.around(np.mean(all_recall, axis=0) / len(json_files), 4) * 100)

        print("Average Recall:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_recall) / len(json_files) * 100, np.std(np.mean(all_recall, axis=1) / len(json_files)) * 100))

        #print("Individual F1:")
        #print(np.around(np.mean(all_f1, axis=0) / len(json_files), 4) * 100)

        print("Average F1:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_f1) / len(json_files) * 100, np.std(np.mean(all_f1, axis=1) / len(json_files)) * 100))