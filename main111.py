from cgi import test
import random
import gc
import os
import numpy as np
from numpy.random import f
from AUC_AUPR import statistic_total_AUC
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from utils import *
from GAT import SEED, Model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import torch.multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from scipy.sparse import vstack, csr_matrix
from sklearn.model_selection import KFold
import shutil
# set_seed(666)
from sklearn import metrics
import warnings
from  train import train_fold
warnings.filterwarnings("ignore", category=Warning)
from zsc_loaddata import Simdata_pro
from zsc_splitdata import split_data
def prepare(param,simData,disease_sim,gene_sim):
    adj_mat = construct_adj_mat(simData['gene_dis_ass']['data_matrix'])
 
    edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long).to(device=param.device)
    het_walk_mat_device = torch.tensor(simData['het_walk_mat']['data_matrix'], dtype=torch.float32).to(device=param.device)
    het_gcn_mat_device = torch.tensor(simData['het_gcn_mat']['data_matrix'], dtype=torch.float32).to(device=param.device)

    adj_df = pd.DataFrame(adj_mat)

    # get_all_samples
    #gene_dis_ass = torch.tensor(simData['gene_dis_ass']['data_matrix'], dtype=torch.float)
    gene_dis_ass = torch.tensor(simData['gene_dis_ass']['data_matrix'], dtype=torch.float)

    #print(f"Model structure: {model}")

    return het_walk_mat_device,het_gcn_mat_device,gene_dis_ass
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.kfolds = 5
        self.batch_size = 256
        self.ratio = 0.2
        self.seed=SEED
        self.Epoch = 60
        self.gcn_layers = 2
        self.head = 8
        self.fm = 128
        self.fd = 128
        self.inSize =128
        self.outSize = 128
        self.nodeNum = 64
        self.hdnDropout = 0.5
        self.fcDropout = 0.5
        self.maskMDI = False
        self.num_u=5032
        self.num_v=5414
        self.lr=0.0005
        self.weight_decay = 0.15
        self.knn_nums=30
        self.label_smoothing=0.05 
        self.gradient_clip_norm = 1.0 
        self.early_stopping_patience = 20
        self.lr_scheduler_patience = 8 
        self.lr_scheduler_factor = 0.7
        self.min_lr = 1e-8 
        self.warmup_epochs = 5 
        self.focal_loss_alpha = 0.3 
        self.focal_loss_gamma = 1.5 
        base_dir ="/root/lanyun-tmp/main/main/result"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        data_split_dir = os.path.join(base_dir, f'数据划分_{timestamp}')
        metrics_dir = os.path.join(base_dir, f'metrics_{timestamp}')
        gradients_dir = os.path.join(base_dir, f'模型_{timestamp}')
        pred_dir = os.path.join(base_dir, f'predictions_{timestamp}')

        for directory in [data_split_dir, metrics_dir, gradients_dir, pred_dir]:
            os.makedirs(directory, exist_ok=True)
        cv_results_file = os.path.join(metrics_dir, f'五折交叉验证结果_{timestamp}.csv')
        with open(cv_results_file, 'w') as f:
            f.write("epoch,fold,auc,aupr,f1_score,accuracy,recall,specificity,precision,test_loss,time(s)\n")
        self.data_split_dir=data_split_dir
        self.metrics_dir=metrics_dir
        self.gradients_dir=gradients_dir
        self.pred_dir= pred_dir
        self.cv_results_file=cv_results_file
        self.datapath='/root/lanyun-tmp/main/main/final data'
        
def main():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param = Config()
    seed = param.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    simData = Simdata_pro(param)
    folds = split_data(param)
    
    dataset=simData
    gene_sim = dataset['gene_walk']['data_matrix']
    disease_sim = dataset['disease_walk']['data_matrix']
    gene_gcnsim = dataset['gene_gcn']['data_matrix']
    dis_gcnsim = dataset['disease_gcn']['data_matrix']
    gene_dis_ass = dataset['gene_dis_ass']['data_matrix']
    if np.sum(diag) != 0:
        disease_sim = disease_sim - np.diag(diag)
    diag = np.diag(gene_sim)
    if np.sum(diag) != 0:
        gene_sim = gene_sim - np.diag(diag)
    diag=np.diag(gene_gcnsim)
    if np.sum(diag) != 0:
        gene_gcnsim = gene_gcnsim - np.diag(diag)
    diag=np.diag(dis_gcnsim)
    if np.sum(diag) != 0:
        dis_gcnsim = dis_gcnsim - np.diag(diag)
    het_walk_mat_device,het_gcn_mat_device,gene_dis_ass=prepare(param,simData,disease_sim,gene_sim)
    
    
    all_fold_aucs, all_fold_model_paths = train_fold(simData, folds, param, het_walk_mat_device, het_gcn_mat_device, gene_dis_ass, state='valid')
    best_idx = np.argmax(all_fold_aucs)
    shutil.copy(all_fold_model_paths[best_idx], os.path.join(param.gradients_dir, 'best_model.pth'))


    all_test_labels = []
    all_test_scores = []

    results_dir = "/root/lanyun-tmp/main/main/result"
    pred_dirs = [d for d in os.listdir(results_dir) if d.startswith('predictions_')]
    
    latest_pred_dir = os.path.join(results_dir, sorted(pred_dirs)[-1])
    print(f"使用最新的预测结果目录: {latest_pred_dir}")

    for fold in range(param.kfolds): 
        pred_file = os.path.join(latest_pred_dir, f'fold_{fold+1}次交叉验证得分顺序.csv')
        print(f"正在读取第{fold+1}折的预测结果: {pred_file}")   
        if os.path.exists(pred_file):   
            fold_pred = pd.read_csv(pred_file)
            all_test_labels.append(fold_pred['true_label'].values)
            all_test_scores.append(fold_pred['predicted_score'].values)
        else:
            print(f"警告: 第{fold+1}折的预测结果文件不存在") 
    if len(all_test_labels) == 5 and len(all_test_scores) == 5:
        print("成功读取所有5折交叉验证的结果")
    else:
        print(f"警告: 只读取到{len(all_test_labels)}折的结果")
    class Args:
        def __init__(self, dataset_name):
            self.dataset = dataset_name         
    args = Args("Gene-Disease Association")
    statistic_total_AUC(args, all_test_labels, all_test_scores)
    
if __name__ == '__main__':
    main()
        