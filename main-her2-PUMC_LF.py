import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from module.extract_excel_features import extract_excel_features
from module.extract_image_features import extract_image_features
from module.combine_features import combine_features
from metrics.plot_roc_curve import plot_roc_curve
from module.inputtotensor import inputtotensor
from metrics.calculate_metrics import calculate_metrics
from module.addbatch import addbatch
from metrics.print_metrics import print_average_metrics, print_mean_std_metrics
from module.set_seed import set_seed
from Classifier.Transformer_Classifier import TransformerClassifier
from module.vis_pre import visualize_predictions
from metrics.plot_dca_curve import plot_dca_curve
from metrics.compute_specificity_fnr import compute_specificity_fnr
# from module import FocalLoss
from module.train_test import train_test
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    # 检查是否已经保存了融合后的特征
    if os.path.exists('HER2_PUMC_LF_combined_features_tensor.pth') and os.path.exists('HER2_PUMC_LF_label_tensor.pth'):
        print("Loading pre-saved features and labels...")
        combined_features_tensor = torch.load('HER2_PUMC_LF_combined_features_tensor.pth')
        label_tensor = torch.load('HER2_PUMC_LF_label_tensor.pth')
    else:
        # Load data
        index, excel_feature, label = extract_excel_features('/tmp/pycharm_project_600/PUMC/HER2_excel_data/HER2-data-特征取交集后汇总.xlsx')
        excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)

        # Extract image features
        image_filenames = ['/tmp/pycharm_project_600/PUMC/HER2_image_data/{}.bmp'.format(idx) for idx in index.astype(int)]
        image_features = extract_image_features(image_filenames)
        pca = PCA(n_components=39)
        image_features_pca = pca.fit_transform(image_features)
        image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)

        # Combine features
        combined_features = combine_features(image_features_pca_tensor, excel_feature_tensor)
        combined_features_tensor, label_tensor = inputtotensor(combined_features, label)

        # 保存融合后的特征和标签
        torch.save(combined_features_tensor, 'HER2_PUMC_LF_combined_features_tensor.pth')
        torch.save(label_tensor, 'HER2_PUMC_LF_label_tensor.pth')

    # # Load data
    # index, excel_feature, label = extract_excel_features('/tmp/pycharm_project_600/PUMC_LF/test.xlsx')
    # excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)
    #
    # # Extract image features
    # image_filenames = ['/tmp/pycharm_project_600/PUMC_LF/US_data/{}.bmp'.format(idx) for idx in index.astype(int)]
    # image_features = extract_image_features(image_filenames)
    # pca = PCA(n_components=99)
    # image_features_pca = pca.fit_transform(image_features)
    # image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)
    #
    # # Combine features
    # combined_features = combine_features(image_features_pca_tensor, excel_feature_tensor)
    # combined_features_tensor, label_tensor = inputtotensor(combined_features, label)
    combined_features = combined_features_tensor.numpy()  # Convert back to numpy for skf.split
    label = label_tensor.numpy()  # Convert back to numpy for skf.split
    print("combined_features_tensor shape:", combined_features_tensor.shape)
    # K-fold cross-validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    all_metrics = {"Validation": [], "Test": []}
    accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro, specificity_scores, FNR_scores = [], [], [], [], [], [], [], []
    all_y_true, all_y_probs = [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features, label):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor[train_index], combined_features_tensor[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]

        # Initialize Transformer model
        net = TransformerClassifier(input_dim=combined_features.shape[1], num_heads=1, num_encoder_layers=1, dim_feedforward=256, output_size=3)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        batch_size = 16
        model_path = f'./pth/best_model_fold{fold}.pth'

        cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred = train_test(
            x_train, y_train, x_test, y_test,
            x_test, y_test,
            net, optimizer, loss_func, batch_size, model_path
        )

        # Metrics calculation
        validation_metrics = calculate_metrics(cm_val, dataset_type="Validation")
        test_metrics = calculate_metrics(cm_test, dataset_type="Test")
        all_metrics["Validation"].append(validation_metrics)
        all_metrics["Test"].append(test_metrics)

        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_test_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_test_pred, average='weighted'))

        # 计算并记录 specificity 和 FNR
        specificity, fnr = compute_specificity_fnr(y_test, test_probs, len(set(label)))
        specificity_scores.append(specificity)
        FNR_scores.append(fnr)

        # ROC curve and AUC for the current fold
        all_y_true.extend(y_test)
        all_y_probs.extend(test_probs)

        plot_dca_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test", save_path=f"DCA_fig/dca_curve_fold{fold}.png")
        roc_auc_fold = plot_roc_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test", save_path=f"ROC_fig/roc_curve_fold{fold}.png")
        AUC_score_macro.append(roc_auc_fold['macro'])
        AUC_score_micro.append(roc_auc_fold['micro'])

    # Compute overall ROC curve and AUC value
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)

    plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall", save_path="ROC_fig/roc_curve.png")
    # overall_roc_auc = plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall")
    # print(overall_roc_auc)

    # Draw overall DAC curve
    plot_dca_curve(all_y_true, all_y_probs, dataset_type="Overall", save_path="DCA_fig/dac_curve.png")
    # overall_dac_metrics = plot_dac_curve(all_y_true, all_y_probs, dataset_type="Overall", save_path="DAC_fig/dac_curve.png")
    # print(overall_dac_metrics)

    print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro, specificity_scores, FNR_scores)



if __name__ == "__main__":
    SEED = 45
    set_seed(SEED)
    main()
