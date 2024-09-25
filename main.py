# ------------------------------------------------------------------------
# Main script used to commence experiments
# ------------------------------------------------------------------------
# Adaption by: Anonymized
# E-Mail: anonymized
# ------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time

import torch
import pandas as pd
import numpy as np
import neptune
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from inertial_baseline.train import run_inertial_network, run_inertial_network_for_features
from utils.torch_utils import fix_random_seed
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt
from camera_baseline.actionformer.main import run_actionformer
from camera_baseline.temporalmaxer.main import run_temporalmaxer
from camera_baseline.tridet.main import run_tridet


def main(args):
    if args.neptune:
        run = neptune.init_run(
        project="",
        api_token=""
        )
    else:
        run = None

    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.neptune:
        run_id = run["sys/id"].fetch()
    else:
        run_id = args.run_id
    
    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', config['name'], str(ts) + '_' + run_id)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()
    
    if args.neptune:
        run['config_name'] = args.config
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))

    if config['name'] == 'tridet' or config['name'] == 'actionformer' or config['name'] == 'temporalmaxer':
        if config['two_stage']:
            orig_feature_folder = config['dataset']['feat_folder']
    
    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True)    

    all_v_pred = np.array([])
    all_v_gt = np.array([])
    all_v_mAP = np.empty((0, len(config['dataset']['tiou_thresholds'])))
        
    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if config['has_null'] == True:
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        config['dataset']['json_anno'] = anno_split
        if 'deepconvlstm' in config['name'] or 'attendanddiscriminate' in config['name'] or 'tinyhar' in config['name']:
            t_losses, v_losses, v_mAP, v_preds, v_gt, net = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
            if config['model']['feature_extract']:
                if net.feature_extract == 'lstm':
                    save_folder = os.path.join('data/{}/processed/lstm_features/'.format(config['dataset_name']), 'sbj_' + str(i))
                elif net.feature_extract == 'attention':
                    save_folder = os.path.join('data/{}/processed/attention_features/'.format(config['dataset_name']), 'sbj_' + str(i))
                run_inertial_network_for_features(net, train_sbjs + val_sbjs, config, save_folder, config['device'])
        elif config['name'] == 'actionformer':
            if config['two_stage']:
                config['dataset']['feat_folder'] = os.path.join(orig_feature_folder, 'sbj_' + str(i))
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'temporalmaxer':
            if config['two_stage']:
                config['dataset']['feat_folder'] = os.path.join(orig_feature_folder, 'sbj_' + str(i))
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_temporalmaxer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'tridet':
            if config['two_stage']:
                config['dataset']['feat_folder'] = os.path.join(orig_feature_folder, 'sbj_' + str(i))
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)

        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + 'sbj_' + str(i) + ' (raw)')
        plt.savefig(os.path.join(log_dir, 'sbj_' + str(i) + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name='sbj_' + str(i) + '_raw')

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline/actionformer/rwhar_loso.yaml')
    parser.add_argument('--run_id', default='', type=str)
    parser.add_argument('--neptune', default=True, type=bool) 
    parser.add_argument('--seed', default=1, type=int)       
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)  

