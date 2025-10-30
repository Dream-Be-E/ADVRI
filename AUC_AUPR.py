import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from scipy import interp
import os
import pandas as pd
from scipy.signal import savgol_filter
def smooth_curve(y, window_length=101, polyorder=3):
      
      if len(y) <= window_length:
          return y  
      
      y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
      y_smooth[0] = y[0]  # 保留起点（FPR=0时的TPR）
      y_smooth[-1] = y[-1]  # 保留终点（FPR=1时的TPR=1）
      return y_smooth

def get_non_overlapping_path(base_path):
   
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path
def statistic_total_AUC(args, KFOLD_test_labels, KFOLD_test_scores, output_dir="/root/lanyun-tmp/main/main/result/数据"):
    print(f"KFOLD_test_labels长度: {len(KFOLD_test_labels)}")
    print(f"KFOLD_test_scores长度: {len(KFOLD_test_scores)}")
    
    
    if isinstance(KFOLD_test_labels, list):
        KFOLD_test_labels = [np.array(labels) for labels in KFOLD_test_labels]
       
    if isinstance(KFOLD_test_scores, list):
        KFOLD_test_scores = [np.array(scores) for scores in KFOLD_test_scores]
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    auc_result = []#
    aupr_result = []

    tprs = []
    precisions = []  
    
    aucs=[]
    mean_fpr = np.linspace(0, 1,10000) 
    #tpr = []
    mean_recall_pr = np.linspace(0, 1, 10000)
   
    for i, (labels, scores) in enumerate(zip(KFOLD_test_labels, KFOLD_test_scores)):
    #for i in range(len(KFOLD_test_labels)):
        '''fpr, tpr, thresholds = roc_curve(KFOLD_test_labels[i], KFOLD_test_scores[i])
        test_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(KFOLD_test_labels[i], KFOLD_test_scores[i])'''
        print(f"\nFold {i+1}:")
        print(f" labels shape: {KFOLD_test_labels[i].shape}")
        
        print(f" scores range: [{np.min(KFOLD_test_scores[i]):.4f}, {np.max(KFOLD_test_scores[i]):.4f}]")
        labels = KFOLD_test_labels[i].flatten()
        scores = KFOLD_test_scores[i].flatten()
        
        # 确保标签是二分类格式（0或1）
        if labels.dtype != np.int64 and labels.dtype != np.int32:
           
            labels = labels.astype(int)
        
        # 检查数据是否有效
        if np.isnan(scores).any() or np.isnan(labels).any():
            print(f"警告: Fold {i+1} 中存在NaN值")
            continue
            
        if len(np.unique(labels)) < 2:
            print(f"警告: Fold {i+1} 中标签只有一个类别")
            continue
        
        # 打印标签的唯一值，用于调试
        unique_labels = np.unique(labels)
        print(f"Fold {i+1} - 标签唯一值: {unique_labels}")
        
      
        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)#不删除中间点
        print(f"Fold {i+1} - 最后5个FPR点: {fpr[-5:]}")
        print(f"Fold {i+1} - 最后5个TPR点: {tpr[-5:]}")
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0#起点0.0
        print(f"  ROC曲线点数: {len(fpr)}")
        print(f"  计算得到的AUC: {roc_auc}")
        tprs.append(interp_tpr)
       
        # 绘制每一折的ROC曲线
        tpr_smooth = smooth_curve(tpr, window_length=50, polyorder=3)
        '''if fpr[-1] < 1.0:
            fpr = np.append(fpr, 1.0)
            tpr_smooth = np.append(tpr_smooth, 1.0)
        else:
            fpr[-1] = 1.0
            tpr_smooth[-1] = 1.0'''
        ax1.plot(fpr, tpr_smooth, lw=1, alpha=0.7, linestyle='--',
                label=f'ROC fold {i+1} (AUC = {roc_auc:.4f})')
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_aupr = auc(recall, precision)
        print(f"  PR曲线点数: {len(precision)}")
        print(f"  计算得到的AUPR: {pr_aupr}")
        #interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        #precisions.append(interp_precision)
        
         # 计算正类比例
        baseline = np.mean(labels)
        # 插入起点 (0,1)
        '''if recall[0] > 0:
            recall = np.insert(recall, 0, 0.0)
            precision = np.insert(precision, 0, 1.0)
        # 插入终点 (1, baseline)
        if recall[-1] < 1:
            recall = np.append(recall, 1.0)
            precision = np.append(precision, baseline)'''
        # 先确保recall是升序的
        order = np.argsort(recall)
        recall = recall[order]
        precision = precision[order]

        # 插值时反向
        interp_precision = np.interp(mean_recall_pr, recall, precision)
        precisions.append(interp_precision)
        # 绘制每一折的PR曲线
        
        precision_smooth = smooth_curve(precision, window_length=51, polyorder=3)
        #ax2.plot(dense_recall, dense_prec, lw=1, alpha=0.3, linestyle='--', 
                   # label=f'PR fold {i+1} (AUPR = {pr_aupr:.4f})')
        # 只绘制recall>0的部分
        mask = recall > 0
        ax2.plot(recall[mask], precision[mask], lw=1, alpha=0.7, linestyle='--', 
                        label=f'PR fold {i+1} (AUPR = {pr_aupr:.4f})')
        auc_result.append(roc_auc)
        aupr_result.append(pr_aupr)
   
    print('\n最终结果:')
    print('-AUC mean: %.4f±%.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          '-AUPR mean: %.4f±%.4f \n' % (np.mean(aupr_result), np.std(aupr_result)))
    # 计算平均ROC曲线
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))#创建两个子图

    mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0#确保最后一个点为1.0
    if len(mean_tpr) > 20:
    # 取倒数第10个点的索引和值
        idx = -20
        tpr_last = mean_tpr[idx]
        # 线性插值：从tpr_last到1.0，共10个点
        mean_tpr[idx:] = np.linspace(tpr_last, 1.0, 20)
# 重新计算平均AUC（确保与插值后的mean_tpr一致）
    mean_auc = auc(mean_fpr, mean_tpr)#计算平均AUC
    std_auc = np.std(aucs)

    # 计算并绘制平均PR曲线
    mean_precision = np.mean(precisions, axis=0)
    mean_precision_smooth = smooth_curve(mean_precision, window_length=51, polyorder=3)
    mean_aupr = np.mean(aupr_result)  # 计算单折AUPR的平均值
    
    #mean_aupr = auc(mean_recall_pr, mean_precision)#平均pr曲线的面积
    std_aupr = np.std(aupr_result)
    #绘制平均曲线与参考线
    mean_tpr_smooth = smooth_curve(mean_tpr, window_length=50, polyorder=3)
    
    '''if mean_fpr[-1] < 1.0:
        mean_fpr = np.append(mean_fpr, 1.0)
        mean_tpr_smooth = np.append(mean_tpr_smooth, 1.0)
    else:
        mean_fpr[-1] = 1.0
        mean_tpr_smooth[-1] = 1.0'''
    # 绘制平滑后的平均ROC曲线
    ax1.plot(mean_fpr, mean_tpr_smooth, color='b', lw=2, alpha=0.8, label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    #ax1.plot(mean_fpr, mean_tpr, color='b',lw=2, alpha=0.8, label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')#linestyle='--', color='r'
    std_tpr = np.std(tpr, axis=0)
    #tpr_upper = np.minimum(mean_tpr + std_tpr, 1)# 计算置信区间上界（无意义）
    #tpr_lower = np.maximum(mean_tpr - std_tpr, 0)# 计算置信区间下界（无意义）
    #ax1.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.') 填充标准差区间（无意义）
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')#添加ROC对角线，用来对比模型
    ax1.set_xlim([-0.05, 1.05])#设置x轴范围
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right')
    
    #mask = mean_recall_pr > 0.01
    #ax2.plot(mean_recall_pr[mask], mean_precision_smooth[mask], color='r', alpha=0.8, label=f'Mean PR (AUPR={mean_aupr:.4f} ± {std_aupr:.4f})')
    all_labels = np.concatenate([arr.flatten() if hasattr(arr, 'flatten') else arr for arr in KFOLD_test_labels])
    baseline = np.mean(all_labels)
# 计算数据集正类比例（PR曲线的基线，随机分类器的Precision等于正类比例）
    #std_precision= np.std(precisions, axis=0)  
    #precision_upper = np.minimum(mean_precision + std_precision, 1)# 计算置信区间上界（无意义）
    #precision_lower = np.maximum(mean_precision - std_precision, 0)# 计算置信区间下界（无意义）
    #ax2.fill_between(mean_recall_pr, precision_lower, precision_upper, color='gray', alpha=0.2)
                     #label='$\pm$ 1 std.dev.') # 填充标准差区间（无意义）
    # 插入起点 (0,1)
    # if mean_recall_pr[0] > 0:
    #     mean_recall_pr = np.insert(mean_recall_pr, 0, 0.0)
    #     mean_precision_smooth = np.insert(mean_precision_smooth, 0, 1.0)
    # 插入终点 (1, baseline)
    if mean_recall_pr[-1] < 1:
        mean_recall_pr = np.append(mean_recall_pr, 1.0)
        mean_precision_smooth = np.append(mean_precision_smooth, baseline)

    mask = (mean_recall_pr > 0.01) & (mean_recall_pr < 0.99)
    mask_mean = mean_recall_pr > 0
    ax2.plot(mean_recall_pr[mask_mean], mean_precision[mask_mean], color='purple', lw=2, 
         label=f'Mean AP: {mean_aupr:.4f} ± {std_aupr:.4f}')
    ax2.axhline(y=baseline, color='gray', linestyle='--')#绘制水平基线（y=正类比例），表示随机分类器的PR曲线，用于对比。
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('PR Curves')
    ax2.legend(loc='lower left')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig_path = os.path.join(output_dir, 'ROC_PR实验.pdf')
    fig_path = get_non_overlapping_path(fig_path)
    plt.savefig(fig_path)
    plt.show()

    # --- 保存所有折的ROC和PR曲线数据为CSV文件 ---
    # 保存每一折的ROC和PR
    for i in range(len(KFOLD_test_labels)):
        labels = KFOLD_test_labels[i].flatten() if hasattr(KFOLD_test_labels[i], 'flatten') else KFOLD_test_labels[i]
        scores = KFOLD_test_scores[i].flatten() if hasattr(KFOLD_test_scores[i], 'flatten') else KFOLD_test_scores[i]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        df_pr = pd.DataFrame({'recall': recall, 'precision': precision})
        roc_path = os.path.join(output_dir, f'ROC_fold_{i+1}.csv')
        pr_path = os.path.join(output_dir, f'PR_fold_{i+1}.csv')
        roc_path = get_non_overlapping_path(roc_path)
        pr_path = get_non_overlapping_path(pr_path)
        df_roc.to_csv(roc_path, index=False)
        df_pr.to_csv(pr_path, index=False)
    # 保存平均曲线
    df_mean_roc = pd.DataFrame({'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr})
    df_mean_pr = pd.DataFrame({'mean_recall': mean_recall_pr, 'mean_precision': mean_precision})
    mean_roc_path = os.path.join(output_dir, 'Mean_ROC.csv')
    mean_pr_path = os.path.join(output_dir, 'Mean_PR.csv')
    mean_roc_path = get_non_overlapping_path(mean_roc_path)
    mean_pr_path = get_non_overlapping_path(mean_pr_path)
    df_mean_roc.to_csv(mean_roc_path, index=False)
    df_mean_pr.to_csv(mean_pr_path, index=False)
    print(f'所有ROC和PR曲线数据已保存到目录: {output_dir}')

def smooth_curve(y, box_pts=20):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
