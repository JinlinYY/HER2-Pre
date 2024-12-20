import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
def plot_roc_curve(y_true, y_probs, dataset_type="Test", save_path=None):
    """
    绘制ROC曲线并打印AUC值
    """
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 20
    # 定义类别名称
    # class_labels = ["LN0", "LN1-3", "LN4+"]
    class_labels = ["HER2-low", "HER2-negative", "HER2-positive"]
    # 将y_true转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均 ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    # 计算微平均 ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 打印AUC值
    print(f"{dataset_type} ROC AUC values:")
    for i in range(n_classes):
        print(f"Class {class_labels[i]} AUC: {roc_auc[i]:.2f}")
    print(f"Macro Average AUC: {roc_auc['macro']:.2f}")
    print(f"Micro Average AUC: {roc_auc['micro']:.2f}")

    # 绘制ROC曲线
    plt.figure(figsize=(9, 6))


    # colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700']
    # colors = ['#B46D43', '#944123', '#384857', '#918B7F']
    # colors = ['#E63946', '#F7A6A4', '#457B9D', '#1D3557']
    colors = ['#AFD0DB', '#2B3956', '#4B7196', '#E8464E']
    # colors = ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D']
    # colors = ['#0A1D37', '#F9A602', '#A4D7E1', '#F0F5F9']

    # colors = ['#e5c8a0', '#9193b4', '#bd9aad', '#d62728', '#9467bd']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4, label=f'{class_labels[i]} ROC curve (AUC = {roc_auc[i]:.2f})')


    # plt.plot(fpr["macro"], tpr["macro"], color='#000080', linestyle='-.', lw=4, label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], color='#E8464E', lw=4,
             label=f'Average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    # plt.plot(fpr["micro"], tpr["micro"], color='#800080', linestyle='--', lw=4, label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_type} ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # 如果提供了保存路径，则保存图片
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png')
        print(f"ROC曲线已保存到: {save_path}")
    else:
        plt.show()

    # 返回计算的 AUC 值字典
    return roc_auc


def plot_roc_curve_2(y_true, y_probs, dataset_type="Test", save_path=None):
    # 确保 y_true 和 y_probs 的正确性
    print(f"Unique values in y_true: {np.unique(y_true)}")  # 应该是 [0, 1]
    print(f"Shape of y_probs: {y_probs.shape}")  # 应该是 (n_samples, 2) 对于二分类

    # 获取类别 1 的预测概率
    y_score = y_probs[:, 1]  # 假设第二列是类别 1 的概率

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, color='#E8464E', lw=4, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color="black", lw=2)  # 对角线
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {dataset_type}')
    plt.legend(loc='lower right')

    # 保存或显示图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"ROC 曲线已保存到: {save_path}")
    else:
        plt.show()

    return roc_auc

