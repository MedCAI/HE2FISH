from dataloader_multilabel import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

import os
import torch
from torch import nn
import torch.optim as optim
import pdb
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

from models.AMIL_MOE import AMIL_MOE
from models.AMIL_complete import MultiscaleMoEAMIL

from utils import *
import csv


class Accuracy_Logger(object):
    """Multi-label Accuracy Logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        # 为每个类别维护 count 和 correct 数量
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, Y_hat, Y):
        """
        Y_hat: 1D array or list of predicted labels (0 or 1), shape: [n_classes]
        Y:     1D array or list of true labels (0 or 1), shape: [n_classes]
        """
        Y_hat = Y_hat[0]
        Y = Y[0]
        for c in range(self.n_classes):
            self.data[c]["count"] += 1
            if Y_hat[c] == Y[c]:
                self.data[c]["correct"] += 1

    def log_batch(self, Y_hat_batch, Y_batch):
        """
        Y_hat_batch: 2D array or list of predicted labels, shape: [batch_size, n_classes]
        Y_batch:     2D array or list of true labels, shape: [batch_size, n_classes]
        """
        Y_hat_batch = np.array(Y_hat_batch).astype(int)
        Y_batch = np.array(Y_batch).astype(int)

        for c in range(self.n_classes):
            self.data[c]["count"] += Y_batch.shape[0]
            self.data[c]["correct"] += (Y_hat_batch[:, c] == Y_batch[:, c]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader), n_classes))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, coords) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(batch_idx)
        data, label = data.to(device), label.to(device)
        label = label[:, 1:]
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, A_raw, results_dict, gate_weights = model(data)
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.cpu().numpy()[0]
        all_preds[batch_idx] = Y_hat.cpu().numpy()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.cpu().numpy()[0]}})
        error = calculate_error(Y_hat, label)
        test_error += error
    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    return patient_results, test_error, auc, acc_logger

if __name__ == "__main__":
    save_dir = './results/Beijing_all/'
    csv_path = './dataset_csv/BH_TCH.csv'
    # data_dir = '/data/ceiling/data/DLBCL/Centers/BH_TCH/uni_scales/'
    data_dir = '/data/ceiling/data/DLBCL/Centers/BH_TCH/uni_scales/'
    weight_dir = './save_weights/Five_5fold_mutation_896_moe_bce555/'
    split_path = '/data/ceiling/workspace/CLAM-master/splits/BH_TCH_mutation/'
    n_classes = 3
    feature_dim = 3072
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                                  data_dir = data_dir,
                                  shuffle = False, 
                                  print_info = True,
                                  label_dict = {0:0, 1:1},
                                  patient_strat=False,
                                  label_col = 'BCL2',
                                  ignore=[])
    os.makedirs(save_dir, exist_ok=True)
    model = MultiscaleMoEAMIL(n_classes=n_classes).cuda()
    folds = [0, 1, 2, 3, 4]
    ckpt_paths = [os.path.join(weight_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
    save_paths = [os.path.join(save_dir, 's_{}_checkpoint.csv'.format(fold)) for fold in folds]
    csv_path = [split_path + 'splits_{}.csv'.format(fold) for fold in folds]
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        # train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=csv_path[ckpt_idx])
        # loader = get_split_loader(test_dataset, testing = False)
        loader = get_simple_loader(dataset)
        model.load_state_dict(torch.load(ckpt_paths[ckpt_idx]))
        patient_results, test_error, auc, acc_logger = summary(model, loader, n_classes=n_classes)
        
        print('test_error: {:.4}, auc: {:.4}'.format(test_error, auc))
        
        for c in range(n_classes):
            print("class_{}:, acc:{:.4}".format(c, acc_logger.get_summary(c)[0]))
            
        with open(os.path.join(save_paths[ckpt_idx]), 'w', newline='') as csvfile:
            fieldnames = ['slide_id', 'prob_BCL2', 'prob_MYC', 'prob_BCL6', 'label_BCL2', 'label_MYC', 'label_BCL6']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in patient_results.items():
                row = {
                    'slide_id': value['slide_id'],
                    'prob_BCL2': value['prob'][0][0],
                    'prob_MYC': value['prob'][0][1],
                    'prob_BCL6': value['prob'][0][2],
                    'label_BCL2': value['label'][0],
                    'label_MYC': value['label'][1],
                    'label_BCL6': value['label'][2],
                }
                writer.writerow(row)
                