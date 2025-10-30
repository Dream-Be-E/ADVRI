import numpy as np
from scipy.sparse import coo_matrix
import os
import torch
import csv
import torch.utils.data.dataset as Dataset
import pandas as pd
from utils import *

def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    # 获取矩阵的维度
    rows, cols = matrix.shape
    
    # 创建边索引列表
    edge_index = []
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:  # 只添加非零元素
                edge_index.append([i, j])
    
    return torch.tensor(edge_index, dtype=torch.long).t()  # 转置以匹配PyTorch的边索引格式


def Simdata_pro(param):
    dataset = dict()
    dis_gcnsim = pd.read_csv("/root/autodl-tmp/autodl-tmp/main/final data/sim_dis_gcn.csv", index_col=0).values
    dis_walksim=pd.read_csv("/root/autodl-tmp/autodl-tmp/main/final data/sim_dis_walk.csv", index_col=0).values
    gene_gcnsim=pd.read_csv("/root/autodl-tmp/autodl-tmp/main/final data/sim_gene_gcn.csv", index_col=0).values
    gene_walksim=pd.read_csv("/root/autodl-tmp/autodl-tmp/main/final data/sim_gene_walk.csv", index_col=0).values
    gene_dis_ass=pd.read_csv("/root/autodl-tmp/autodl-tmp/main/final data/GDassociation_matrix_T.csv", index_col=0).values
    '''dis_walksim = pd.read_csv("D:\\1_awork\\data\\final data\\sim_dis_walk.csv", index_col=0).values
    gene_walksim=pd.read_csv("D:\\1_awork\data\\final data\sim_gene_walk.csv", index_col=0).values
    dis_gcnsim=pd.read_csv("D:\\1_awork\data\\final data\sim_dis_gcn.csv", index_col=0).values
    gene_gcnsim=pd.read_csv("D:\\1_awork\data\\final data\sim_gene_gcn.csv", index_col=0).values
    gene_dis_ass =pd.read_csv("D:\\1_awork\data\\final data\GDassociation_matrix_T.csv",index_col=0).values#基因疾病关联矩阵'''
    #gene_gene=pd.read_csv("D:\\1_awork\data\\final data\sim_gene_walk.csv").values
    #disease_disease=pd.read_csv("D:\\1_awork\data\\final data\sim_dis_walk.csv").values
    'gene-gene'
    #gene_gene_edge_index = get_edge_index(gene_gene)
    #dataset['gene_gene'] = {'data_matrix': gene_gene, 'edges': gene_gene_edge_index}
    'DIS-DIS'
    #disease_disease_edge_index = get_edge_index(disease_disease)
    #dataset['disease_disease'] = {'data_matrix': disease_disease, 'edges': disease_disease_edge_index}

    "gene-walk-sim"
    gene_walksim_edge_index = get_edge_index(gene_walksim)
    dataset['gene_walk'] = {'data_matrix': gene_walksim, 'edges': gene_walksim_edge_index}
    "disease walk sim"
    dis_walksim_edge_index = get_edge_index(dis_walksim)
    dataset['disease_walk'] = {'data_matrix': dis_walksim, 'edges': dis_walksim_edge_index}
    "gene_gcn sim"
    gene_gcnsim_edge_index = get_edge_index(gene_gcnsim)
    dataset['gene_gcn'] = {'data_matrix':gene_gcnsim, 'edges':gene_gcnsim_edge_index}
    "disease gcn sim"
    dis_gcnsim_edge_index = get_edge_index(dis_gcnsim)
    dataset['disease_gcn'] = {'data_matrix': dis_gcnsim, 'edges': dis_gcnsim_edge_index}
    "gene_dis_ass sim"
    gene_dis_ass_edge_index = get_edge_index(gene_dis_ass)
    dataset['gene_dis_ass'] = {'data_matrix': gene_dis_ass, 'edges': gene_dis_ass_edge_index}

    het_walk_mat = construct_het_mat(gene_dis_ass, dis_walksim,gene_walksim)
    het_gcn_mat = construct_het_mat(gene_dis_ass, dis_gcnsim,gene_gcnsim)
    
    '异构矩阵边索引'
    hetwalk_edge_index = get_edge_index(het_walk_mat)
    dataset['het_walk_mat'] = {'data_matrix': het_walk_mat, 'edges': hetwalk_edge_index}
    
    hetgcn_edge_index = get_edge_index(het_gcn_mat)
    dataset['het_gcn_mat'] = {'data_matrix': het_gcn_mat, 'edges': hetgcn_edge_index}
    
    return dataset


class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):

        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label