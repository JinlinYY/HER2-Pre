from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dca_curve(y_true, y_probs, dataset_type="Test", save_path=None):
    thresholds = np.linspace(0, 1, 101)

    # 将 y_true 转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]
    print(n_classes)

    # Initialize lists to store net benefit and benefits
    net_benefits = []  # Store Net Benefit for each threshold
    treat_all_benefits = []  # Store Treat All benefits
    treat_none_benefits = []  # Store Treat None benefits

    # Iterate over thresholds
    for threshold in thresholds:
        total_tp = total_fp = total_tn = total_fn = 0  # Initialize counters for TP, FP, TN, FN

        # Iterate through each class to calculate net benefit
        for i in range(n_classes):  # For each class
            # Generate predicted classes based on the current threshold
            y_pred = (y_probs[:, i] >= threshold).astype(int)

            # Compute confusion matrix for each class (treat it as positive)
            cm = confusion_matrix(y_true_binarized[:, i], y_pred)  # Binarized y_true vs. predicted class

            # For each class, compute the confusion matrix and update totals
            if cm.shape == (2, 2):  # Ensure that it's a 2x2 confusion matrix for binary classification
                tn, fp, fn, tp = cm.ravel()

                # Update totals for Net Benefit calculation
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

        # Compute Net Benefit for the current threshold
        if threshold < 1:  # Avoid division by zero for (1 - p)
            p = threshold  # Threshold value
            net_benefit = (total_tp / len(y_true)) - (total_fp / len(y_true)) * (p / (1 - p))
            net_benefits.append(net_benefit)
        else:
            net_benefits.append(np.nan)  # Append NaN for invalid thresholds

        # Compute Treat All Benefit (based on the proportion of true positives)
        if np.sum(y_true_binarized[:, 1]) > 0:  # Avoid division by zero
            treat_all_benefit = (total_tp / len(y_true)) - (total_tn / len(y_true)) * (
                        p / (1 - p)) if threshold < 1 else np.nan
            treat_all_benefits.append(treat_all_benefit)
        else:
            treat_all_benefits.append(np.nan)  # Append NaN if no positives exist

        # Compute Treat None Benefit (always 0)
        treat_none_benefits.append(0)

    # Convert lists to arrays and handle NaN values
    net_benefits = np.nan_to_num(net_benefits, nan=0)
    treat_all_benefits = np.nan_to_num(treat_all_benefits, nan=0)

    # Find all intersection points
    intersections = []
    for i in range(1, len(thresholds)):
        # Check if the sign of the difference changes (from positive to negative or vice versa)
        if (net_benefits[i-1] - treat_all_benefits[i-1]) * (net_benefits[i] - treat_all_benefits[i]) < 0:
            intersection_threshold = (thresholds[i-1] + thresholds[i]) / 2  # Approximate intersection threshold
            intersection_value = (net_benefits[i-1] + net_benefits[i]) / 2  # Approximate intersection value
            intersections.append((intersection_threshold, intersection_value))

    # Plot Net Benefit, Treat All, and Treat None using plt directly
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, net_benefits, label='Net Benefit', color='#E8464E', lw=4)
    plt.plot(thresholds, treat_all_benefits, label='Treat All', color='#2B3956', linestyle='--', lw=4)
    plt.plot(thresholds, treat_none_benefits, label='Treat None', color='#4B7196', linestyle='--', lw=4)

    # Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(treat_all_benefits, 0)
    y1 = np.maximum(net_benefits, y2)
    plt.fill_between(thresholds, y1, y2, color='#2B3956', alpha=0.2)

    # Annotate all intersection points
    for intersection in intersections:
        plt.scatter(intersection[0], intersection[1], color='purple', zorder=8)
        plt.text(intersection[0] + 0.02, intersection[1], f'Threshold = {intersection[0]:.2f}',
                 color='purple', fontsize=16)

    plt.title(f'{dataset_type} Net Benefit')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.xlim([-0.05, 1.05])  # Set x-axis limits
    plt.ylim([min(net_benefits) - 0.05, max(net_benefits) + 0.05])  # Set y-axis limits
    plt.legend(loc='upper right')

    # Save or display the plot
    if save_path:
        net_benefit_path = os.path.splitext(save_path)[0] + "_net_benefit.png"
        plt.savefig(net_benefit_path, format='png', bbox_inches='tight')
        print(f"净效益曲线已保存到: {net_benefit_path}")
    else:
        plt.show()

    # Return calculated metrics (just net benefits and thresholds)
    return {
        "thresholds": thresholds,
        "net_benefit": net_benefits,
        "treat_all_benefit": treat_all_benefits,
        "treat_none_benefit": treat_none_benefits,
        "intersections": intersections  # Return all intersection points
    }
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


def plot_dca_curve_2(y_true, y_probs, dataset_type="Test", save_path=None):
    # 确保 y_true 是 NumPy 数组或 PyTorch 张量
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()  # 如果是 PyTorch 张量，转换为 NumPy 数组

    # 调试：检查 y_true 和 y_probs
    print("Unique values in y_true:", np.unique(y_true))  # 确保 y_true 只有 0 和 1
    print("Shape of y_probs:", y_probs.shape)  # 确保 y_probs 是 (n_samples, 2)

    # 确保 y_probs 只包含与 y_true 对应的概率
    y_probs_filtered = y_probs  # 由于是二分类，直接使用 y_probs

    # 计算类别数量
    n_classes = len(np.unique(y_true))  # 直接从 y_true 计算类别数
    print("Number of classes:", n_classes)  # 调试：打印类别数量

    assert n_classes == 2, "This function is designed for binary classification only."

    # 初始化存储净效益和其他益处的列表
    net_benefits = []  # 存储每个阈值的净效益
    treat_all_benefits = []  # 存储 Treat All 的益处
    treat_none_benefits = []  # 存储 Treat None 的益处

    # 遍历每个阈值
    thresholds = np.linspace(0, 1, 101)
    for threshold in thresholds:
        total_tp = total_fp = total_tn = total_fn = 0  # 初始化 TP、FP、TN 和 FN 计数器

        # 计算混淆矩阵和净效益
        y_pred = (y_probs_filtered[:, 1] >= threshold).astype(int)  # 使用类别1的概率作为预测

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # 确保是二分类混淆矩阵
            tn, fp, fn, tp = cm.ravel()
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        # 计算当前阈值下的净效益
        if threshold < 1:
            p = threshold
            net_benefit = (total_tp / len(y_true)) - (total_fp / len(y_true)) * (p / (1 - p))
            net_benefits.append(net_benefit)
        else:
            net_benefits.append(np.nan)

        # 计算 Treat All 益处
        if np.sum(y_true == 1) > 0:  # 避免除以零
            treat_all_benefit = (total_tp / len(y_true)) - (total_tn / len(y_true)) * (
                        p / (1 - p)) if threshold < 1 else np.nan
            treat_all_benefits.append(treat_all_benefit)
        else:
            treat_all_benefits.append(np.nan)

        # 计算 Treat None 益处（始终为 0）
        treat_none_benefits.append(0)

    # 处理 NaN 值
    net_benefits = np.nan_to_num(net_benefits, nan=0)
    treat_all_benefits = np.nan_to_num(treat_all_benefits, nan=0)

    # 找到所有交点
    intersections = []
    for i in range(1, len(thresholds)):
        if (net_benefits[i - 1] - treat_all_benefits[i - 1]) * (net_benefits[i] - treat_all_benefits[i]) < 0:
            intersection_threshold = (thresholds[i - 1] + thresholds[i]) / 2  # 近似交点阈值
            intersection_value = (net_benefits[i - 1] + net_benefits[i]) / 2  # 近似交点值
            intersections.append((intersection_threshold, intersection_value))

    # 绘制 Net Benefit, Treat All 和 Treat None 曲线
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, net_benefits, label='Net Benefit', color='#E8464E', lw=4)
    plt.plot(thresholds, treat_all_benefits, label='Treat All', color='#2B3956', linestyle='--', lw=4)
    plt.plot(thresholds, treat_none_benefits, label='Treat None', color='#4B7196', linestyle='--', lw=4)

    # 填充 Net Benefit 和 Treat All 曲线之间的区域
    y2 = np.maximum(treat_all_benefits, 0)
    y1 = np.maximum(net_benefits, y2)
    plt.fill_between(thresholds, y1, y2, color='#2B3956', alpha=0.2)

    # 标注所有交点
    for intersection in intersections:
        plt.scatter(intersection[0], intersection[1], color='purple', zorder=8)
        plt.text(intersection[0] + 0.02, intersection[1], f'Threshold = {intersection[0]:.2f}',
                 color='purple', fontsize=16)

    plt.title(f'{dataset_type} Net Benefit')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.xlim([-0.05, 1.05])
    plt.ylim([min(net_benefits) - 0.05, max(net_benefits) + 0.05])
    plt.legend(loc='upper right')

    # 保存或显示图表
    if save_path:
        net_benefit_path = os.path.splitext(save_path)[0] + "_net_benefit.png"
        plt.savefig(net_benefit_path, format='png', bbox_inches='tight')
        print(f"净效益曲线已保存到: {net_benefit_path}")
    else:
        plt.show()

    # 返回计算的指标
    return {
        "thresholds": thresholds,
        "net_benefit": net_benefits,
        "treat_all_benefit": treat_all_benefits,
        "treat_none_benefit": treat_none_benefits,
        "intersections": intersections
    }
