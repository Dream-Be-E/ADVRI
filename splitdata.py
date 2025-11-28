import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split

def split_data(param):
    ratio = param.ratio
    gd_matrix = pd.read_csv(os.path.join(param.datapath+'/GDassociation_matrix_T.csv'), dtype=int, delimiter=',',index_col=0)
    gd_matrix.index = gd_matrix.index.astype(int)  
    gd_matrix.columns = gd_matrix.columns.astype(int)  
    disease_ids = gd_matrix.index.tolist()   
    gene_ids = gd_matrix.columns.tolist()  
    # get the edge of positives samples
    rng = np.random.default_rng(seed=42) 
    '''pos_samples = np.where(gd_matrix == 1)#正样本,得到两个数组，一个基因，一个疾病
    print(pos_samples[1].shape[0])
    print("正样本数量：", pos_samples[0].shape[0])  # 应该等于pos_samples[1].shape[0]
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)'''

    # get the edge of negative samples
    '''rng = np.random.default_rng(seed=42)
    neg_samples = np.where(gd_matrix == 0)
    print("正样本数量：", neg_samples[0].shape[0])  # 应该等于pos_samples[1].shape[0]
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]'''

    
    def balance_pos_neg(edges, labels, random_state=42):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos = len(pos_idx)
        np.random.seed(random_state)
        if n_pos == 0:
            raise ValueError("No positive samples found!")
        if len(neg_idx) < n_pos:
            raise ValueError("Not enough negative samples to balance!")
        neg_idx_sampled = np.random.choice(neg_idx, n_pos, replace=False)
        sel_idx = np.concatenate([pos_idx, neg_idx_sampled])
        np.random.shuffle(sel_idx)
        return edges[sel_idx], labels[sel_idx]
    #pos_samples = (np.array(pos_gene_ids), np.array(pos_disease_ids))  # 替换原pos_samples
    pos_dis_idx, pos_gene_idx = np.where(gd_matrix.values == 1)
    pos_disease_ids = np.array([disease_ids[d] for d in pos_dis_idx], dtype=int) # 列号c → 疾病ID=disease_ids[c]
    pos_gene_ids = np.array([gene_ids[g] for g in pos_gene_idx], dtype=int)
    pos_edges = np.vstack((pos_disease_ids, pos_gene_ids)).T
    pos_labels = np.ones(pos_edges.shape[0], dtype=int)
    # 负样本：同理转换
    neg_dis_idx, neg_gene_idx = np.where(gd_matrix.values == 0)
    neg_disease_ids = np.array([disease_ids[d] for d in neg_dis_idx], dtype=int)
    neg_gene_ids = np.array([gene_ids[g] for g in neg_gene_idx], dtype=int)
    neg_edges = np.vstack((neg_disease_ids, neg_gene_ids)).T
    neg_labels = np.zeros(neg_edges.shape[0], dtype=int)
    edge_idx_dict = dict()

    '''test_pos_edges = pos_samples_shuffled[:, :idx_split]#测试集正样本边
    test_neg_edges = neg_samples_shuffled[:, :idx_split]#测试集负样本边
    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))#将正负样本边垂直堆叠
    # np.savetxt('./train_test/test_pos.csv', test_pos_edges, delimiter=',')
    # np.savetxt('./train_test/test_neg.csv', test_neg_edges, delimiter=',')'''



    '''train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))'''


    ''' edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    edge_idx_dict['true_md'] = gd_matrix'''##*
    save_dir = os.path.join(param.data_split_dir, "5fold_data")
    os.makedirs(save_dir, exist_ok=True)
    #return edge_idx_dict

    all_edges = np.vstack((pos_edges, neg_edges))
    all_labels = np.concatenate((pos_labels, neg_labels))
    train_test_edges, val_edges, train_test_labels, val_labels = train_test_split(
        all_edges, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    train_test_edges = np.array(train_test_edges)
    train_test_labels = np.array(train_test_labels)
    val_edges = np.array(val_edges)
    val_labels = np.array(val_labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    
    
    for fold, (train_index, test_index) in enumerate(skf.split(train_test_edges, train_test_labels)):
        fold_dict = {}

        train_edges = train_test_edges[train_index]
        train_labels = train_test_labels[train_index]
        test_edges = train_test_edges[test_index]
        test_labels = train_test_labels[test_index]
        # 正负均衡采样
        train_edges_bal, train_labels_bal = balance_pos_neg(train_edges, train_labels, random_state=fold+100)
        test_edges_bal, test_labels_bal = balance_pos_neg(test_edges, test_labels, random_state=fold+200)
        fold_dict['train_Edges'] = train_edges_bal
        fold_dict['train_Labels'] = train_labels_bal
        fold_dict['test_Edges'] = test_edges_bal
        fold_dict['test_Labels'] = test_labels_bal
        fold_dict['val_Edges'] = val_edges
        fold_dict['val_Labels'] = val_labels
        fold_dict['true_md'] = gd_matrix
        folds.append(fold_dict)
        
        # 保存训练集
        #train_df = pd.DataFrame(fold_dict['train_Edges'], columns=['disease_id', 'gene_id'])
        #train_df['label'] = fold_dict['train_Labels']
        #train_df.to_csv(os.path.join(save_dir, f'fold_{fold+1}_train.csv'), index=False)

        # 保存测试集
        #test_df = pd.DataFrame(fold_dict['test_Edges'], columns=['disease_id', 'gene_id'])
        #test_df['label'] = fold_dict['test_Labels']
        #test_df.to_csv(os.path.join(save_dir, f'fold_{fold+1}_test.csv'), index=False)

        train_pos = np.sum(fold_dict['train_Labels'] == 1)
        train_neg = np.sum(fold_dict['train_Labels'] == 0)
        test_pos = np.sum(fold_dict['test_Labels'] == 1)
        test_neg = np.sum(fold_dict['test_Labels'] == 0)
        #print(f"Fold {fold+1}:")
        #print(f"  Train set: {len(fold_dict['train_Labels'])} samples (pos: {train_pos}, neg: {train_neg})")
        
        #print(f"  Test  set: {len(fold_dict['test_Labels'])} samples (pos: {test_pos}, neg: {test_neg})")
        #print("-" * 40)
    return folds