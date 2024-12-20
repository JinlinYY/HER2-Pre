from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import numpy as np

def compute_specificity_fnr(y_true, y_probs, n_classes, threshold=0.5):
    # 将 y_true 转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))

    # Initialize dictionaries to store specificity and FNR for each class
    specificities = []
    fnrs = []

    # 对预测结果进行阈值处理，将概率转化为标签
    y_pred_binarized = (y_probs >= threshold).astype(int)

    # Calculate specificity and FNR for each class
    for i in range(n_classes):
        cm = confusion_matrix(y_true_binarized[:, i], y_pred_binarized[:, i])

        if cm.shape == (2, 2):  # Ensure it is a 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()

            # Specificity = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            # False Negative Rate = FN / (FN + TP)
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

            specificities.append(specificity)
            fnrs.append(fnr)

    return np.mean(specificities), np.mean(fnrs)




###################二分类###########################

# def compute_specificity_fnr_2(y_true, y_probs):
#     # 对预测结果进行阈值处理，将概率转化为标签
#     # y_pred_binarized = (y_probs >= threshold).astype(int)
#
#     # 计算混淆矩阵
#     cm = confusion_matrix(y_true, y_probs)
#
#     if cm.shape == (2, 2):  # Ensure it is a 2x2 confusion matrix
#         tn, fp, fn, tp = cm.ravel()
#
#         # Specificity = TN / (TN + FP)
#         specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
#
#         # False Negative Rate = FN / (FN + TP)
#         fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
#
#         return specificity, fnr
#     else:
#         raise ValueError("Confusion matrix should be of shape (2, 2) for binary classification.")


def compute_specificity_fnr_2(y_true, y_probs):
    # 将概率转换为离散标签
    y_pred = np.argmax(y_probs, axis=1)  # 假设 y_probs 是一个形状为 [n_samples, n_classes] 的概率矩阵

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算 specificity 和 FNR（你可以根据需求实现这部分）
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # True Negative Rate
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])  # False Negative Rate

    return specificity, fnr