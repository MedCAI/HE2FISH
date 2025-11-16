from dataloader_clinic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

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

from models.AMIL_clinic import MultiscaleMoEAMIL, ModelWithClinic
from utils import *
from ASL_Loss import AsymmetricLoss, FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_mask(data, r, p):
    """
    随机mask掉r%的实例，保留(1-r)%的实例，只有在p的概率下才执行mask操作。
    :param data: 输入的张量，形状为 (N, dim)
    :param r: 掩盖的比例，例如r = 0.2表示mask掉20%的实例
    :param p: 执行mask操作的概率，例如p = 0.5表示50%的概率执行mask
    :return: 保留(1-r)%的实例，形状为 ((1-r)N, dim)，如果没有执行mask操作，则返回原数据。
    """
    N = data.size(0)
    
    # 以概率 p 决定是否执行 mask 操作
    if torch.rand(1).item() < p:
        # 生成一个随机的 mask，随机选择哪些样本保留
        mask = torch.rand(N) > r  # 返回一个长度为 N 的布尔向量，True表示保留该样本，False表示掩盖
        
        # 如果所有的样本都被mask掉（即mask的sum为0），强制保留至少1个样本
        if mask.sum() == 0:
            mask[0] = True  # 确保至少保留第一个样本
        
        # 根据 mask 筛选样本
        masked_data = data[mask]
        return masked_data
    else:
        # 如果没有执行mask操作，直接返回原数据
        return data
    
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
    model[0].eval()
    model[1].eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, clinic) in enumerate(loader):
        data, label, clinic = data.to(device), label.to(device), clinic.to(device)
        label = label[:, 1:4]
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            image_features = model[0](data)
            logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list = model[1]([image_features, clinic])
        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.cpu().numpy()[0]
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.cpu().numpy()[0]}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 1:
        auc = roc_auc_score(all_labels, all_probs[:, 0])
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

def train_abmil(datasets,
                save_path='./save_weights/camelyon16_transmil_imagenet/',
                feature_dim = 1536,
                n_classes = 2,
                fold = 0,
                writer_flag = True,
                max_epoch = 200,
                early_stopping = True,
                ):
    writer_dir = os.path.join(save_path, str(fold))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir)
    if writer_flag:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    
    print("\nInit train/val/test splits...")
    train_split, val_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print("\nInit loss function...")
    loss_fn = FocalLoss(alpha=0.9, gamma=5)
    # pos_weight = torch.tensor([10.0, 10.0, 10.0]).cuda()
    # loss_fn =  nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    image_extractor = MultiscaleMoEAMIL(n_classes=n_classes)
    image_extractor.load_state_dict(torch.load('./save_weights/BH_TCH_5fold_mutation_896_moe_bce555/s_{}_checkpoint_10.pt'.format(fold)))
    _ = image_extractor.to(device)
    clinic_model = ModelWithClinic()
    _ = clinic_model.to(device)
    model = [image_extractor, clinic_model]
    
    print("\nInit optimizer")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, clinic_model.parameters()), lr=1e-4, weight_decay=1e-5)
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, weighted = False)
    val_loader = get_split_loader(val_split,  testing = False)
    test_loader = get_split_loader(test_split, testing = False)
    print('Done!')
    
    mini_loss = np.Inf
    retain = 0
    for epoch in range(max_epoch):
        train_loop(epoch, model, train_loader, optimizer, n_classes, writer, loss_fn)
        loss = validate(epoch, model, val_loader, n_classes, writer, loss_fn)
        if epoch % 1 == 0:
            torch.save(model[1].state_dict(), os.path.join(save_path, 's_{}_checkpoint_{}.pt'.format(fold, epoch)))
        if loss < mini_loss:
            print("loss decrease from:{} to {}".format(mini_loss, loss))
            torch.save(model[1].state_dict(), os.path.join(save_path, 's_{}_checkpoint.pt'.format(fold)))
            mini_loss = loss
            retain = 0
        else:
            retain += 1
            print("Retain of early stopping: {} / {}".format(retain, 30))
        if early_stopping:
            # if retain > 30 and epoch > 20:
            if epoch >= 5:
                print("Early stopping")
                break
    model[1].load_state_dict(torch.load(os.path.join(save_path, 's_{}_checkpoint.pt'.format(fold))))
    _, val_error, val_auc, _= summary(model, val_loader, n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

def train_loop(epoch, model, loader, optimizer, n_classes, writer, loss_fn):
    model[0].eval()
    model[1].train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    bag_loss = 0.
    
    print('\n')
    accumulation_steps = 8
    for batch_idx, (data, label, clinic) in enumerate(loader):
        data, label, clinic = data.to(device), label.to(device), clinic.to(device)
        if data.size(0) > 120000:
            continue
        data = random_mask(data, r=0.5, p=0.8)
        label = label[:, 1:4]
        with torch.no_grad():
            image_features = model[0](data)
        logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list = model[1]([image_features, clinic])
        acc_logger.log(Y_hat, label)
    
        loss_bag = loss_fn(logits, label)
    
        loss = loss_bag
        loss_bag_value = loss_bag.item()
        loss_value = loss.item()
        
        train_loss += loss_value
        bag_loss += loss_bag
    
        # Normalize loss to account for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
    
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, loss_bag: {:.4f}, label: {}, bag_size: {}'.format(
                batch_idx, loss_value, loss_bag_value, label.cpu().numpy()[0], data.size(0)
            ))
    
    # 注意：如果最后剩下不足 accumulation_steps 的 batch，要再 step 一下
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= len(loader)
    bag_loss /= len(loader)
    
    print('Epoch: {}, train_loss: {:.4f}, bag_loss: {:.4f}, '.format(epoch, train_loss, bag_loss))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/loss_bag', loss_bag, epoch)


def validate(epoch, model, loader, n_classes, writer, loss_fn):
    model[0].eval()
    model[1].eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))
    
    with torch.no_grad():
        for batch_idx, (data, label, clinic) in enumerate(loader):
            data, label, clinic = data.to(device, non_blocking=True), label.to(device, non_blocking=True), clinic.to(device, non_blocking=True)
            if data.size(0) > 120000:
                continue
            label = label[:, 1:4]
            image_features = model[0](data)
            logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list = model[1]([image_features, clinic])
            acc_logger.log(Y_hat, label)
            loss_bag = loss_fn(logits, label)
            loss = loss_bag

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.cpu().numpy()[0]
            
            val_loss += loss.item()

    val_loss /= len(loader)
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
    print('\nVal Set, val_loss: {:.4f}, auc: {:.4f}'.format(val_loss, auc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    return val_loss

if __name__ == "__main__":
    # csv_path = './dataset_csv/BH_TCH.csv'
    # data_dir = '/data/ceiling/data/DLBCL/Centers/BH_TCH/uni_scales/'
    # split_path = '/data/ceiling/workspace/CLAM-master/splits/BH_TCH_mutation/'
    # save_dir = './save_weights/BH_TCH_5fold_mutation_896_moe_bce555_fold4/'

    csv_path = './dataset_csv/TCH_SXH_merged.csv'
    data_dir = '/data/ceiling/data/DLBCL/Centers/BH_TCH/uni_scales/'
    split_path = '/data/ceiling/workspace/CLAM-master/splits/Clinic/'
    save_dir = './save_weights/Clinic_5fold_mutation_896_moe_focal/'
    
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                                  data_dir = data_dir,
                                  shuffle = False, 
                                  seed = 4, 
                                  print_info = True,
                                  label_dict = {0:0, 1:1},
                                  patient_strat=False,
                                  label_col = 'BCL2',
                                  ignore=[])
    
    csv_path = [split_path + 'splits_{}.csv'.format(i) for i in range(5)]
    # csv_path = [split_path + 'splits_{}.csv'.format(i) for i in range(4, 5)]
    for step, name in enumerate(csv_path):
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=name)
        train_abmil((train_dataset, val_dataset, test_dataset),
                     save_path=save_dir,
                     feature_dim = 3072,
                     n_classes = 3,
                     fold = step,
                     writer_flag = True,
                     max_epoch = 200,
                     early_stopping = True,)