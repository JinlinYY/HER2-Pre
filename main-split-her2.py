import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from module.extract_excel_features import extract_excel_features
from module.extract_image_features import extract_image_features
from module.combine_features import combine_features
from metrics.plot_roc_curve import plot_roc_curve
from module.inputtotensor import inputtotensor
from metrics.calculate_metrics import calculate_metrics
from module.addbatch import addbatch
from metrics.print_metrics import print_average_metrics
from module.set_seed import set_seed
from Classifier.Transformer_Classifier import TransformerClassifier
from module.vis_pre import visualize_predictions
from metrics.plot_dca_curve import plot_dca_curve
from metrics.compute_specificity_fnr import compute_specificity_fnr
from module.train_test import train_test
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams['font.size'] = 18

    # 检查是否已经保存了融合后的特征
    if os.path.exists('HER2_combined_train_features_tensor.pth') and os.path.exists('HER2_train_label_tensor.pth') and \
            os.path.exists('HER2_combined_val_features_tensor.pth') and os.path.exists('HER2_val_label_tensor.pth') and \
            os.path.exists('HER2_combined_test_features_tensor.pth') and os.path.exists('HER2_test_label_tensor.pth'):
        print("Loading pre-saved features and labels...")
        combined_train_features_tensor = torch.load('HER2_combined_train_features_tensor.pth')
        train_label_tensor = torch.load('HER2_train_label_tensor.pth')
        combined_val_features_tensor = torch.load('HER2_combined_val_features_tensor.pth')
        val_label_tensor = torch.load('HER2_val_label_tensor.pth')
        combined_test_features_tensor = torch.load('HER2_combined_test_features_tensor.pth')
        test_label_tensor = torch.load('HER2_test_label_tensor.pth')
    else:
        # Load training data (Excel)
        train_index, train_excel_feature, train_label = extract_excel_features(
            '/tmp/pycharm_project_600/PUMC_LF/编码后汇总-协和+隆福-HER2表达分类-split_data71515-train-del.xlsx')
        train_excel_feature_tensor = torch.tensor(train_excel_feature, dtype=torch.float32)

        # Load validation data (Excel)
        val_index, val_excel_feature, val_label = extract_excel_features(
            '/tmp/pycharm_project_600/PUMC_LF/编码后汇总-协和+隆福-HER2表达分类-split_data71515-val-del.xlsx')
        val_excel_feature_tensor = torch.tensor(val_excel_feature, dtype=torch.float32)

        # Load test data (Excel)
        test_index, test_excel_feature, test_label = extract_excel_features(
            '/tmp/pycharm_project_600/PUMC_LF/编码后汇总-协和+隆福-HER2表达分类-split_data71515-test-del.xlsx')
        test_excel_feature_tensor = torch.tensor(test_excel_feature, dtype=torch.float32)

        # Extract image features (all images in one folder)
        all_image_filenames = ['/tmp/pycharm_project_600/PUMC_LF/US_data/{}.bmp'.format(idx) for idx in
                               np.concatenate([train_index, val_index, test_index])]

        all_image_features = extract_image_features(all_image_filenames)

        # PCA for image features
        pca = PCA(n_components=39)
        all_image_features_pca = pca.fit_transform(all_image_features)
        all_image_features_pca_tensor = torch.tensor(all_image_features_pca, dtype=torch.float32)

        # Combine features (Excel features + Image features)
        combined_train_features = combine_features(
            torch.tensor(all_image_features_pca[:len(train_index)], dtype=torch.float32), train_excel_feature_tensor)
        combined_val_features = combine_features(
            torch.tensor(all_image_features_pca[len(train_index):len(train_index) + len(val_index)],
                         dtype=torch.float32), val_excel_feature_tensor)
        combined_test_features = combine_features(
            torch.tensor(all_image_features_pca[len(train_index) + len(val_index):], dtype=torch.float32),
            test_excel_feature_tensor)

        combined_train_features_tensor, train_label_tensor = inputtotensor(combined_train_features, train_label)
        combined_val_features_tensor, val_label_tensor = inputtotensor(combined_val_features, val_label)
        combined_test_features_tensor, test_label_tensor = inputtotensor(combined_test_features, test_label)

        # 保存融合后的特征和标签
        torch.save(combined_train_features_tensor, 'HER2_combined_train_features_tensor.pth')
        torch.save(train_label_tensor, 'HER2_train_label_tensor.pth')
        torch.save(combined_val_features_tensor, 'HER2_combined_val_features_tensor.pth')
        torch.save(val_label_tensor, 'HER2_val_label_tensor.pth')
        torch.save(combined_test_features_tensor, 'HER2_combined_test_features_tensor.pth')
        torch.save(test_label_tensor, 'HER2_test_label_tensor.pth')

    # 训练集、验证集、测试集
    x_train, x_val, x_test = combined_train_features_tensor, combined_val_features_tensor, combined_test_features_tensor
    y_train, y_val, y_test = train_label_tensor, val_label_tensor, test_label_tensor
    print("combined_features_tensor shape:", combined_train_features_tensor.shape)
    # Initialize Transformer model
    net = TransformerClassifier(input_dim=combined_train_features_tensor.shape[1], num_heads=5, num_encoder_layers=1,
                                dim_feedforward=256, output_size=3)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    batch_size = 16
    model_path = f'./pth/best_model.pth'

    # Train and evaluate model
    cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred = train_test(
        x_train, y_train, x_val, y_val,
        x_test, y_test,
        net, optimizer, loss_func, batch_size, model_path
    )

    # Metrics calculation for validation and test sets
    validation_metrics = calculate_metrics(cm_val, dataset_type="Validation")
    test_metrics = calculate_metrics(cm_test, dataset_type="Test")

    # Validation metrics
    accuracy_score_val = accuracy_score(y_val, y_val_pred)
    precision_score_val = precision_score(y_val, y_val_pred, average='weighted')
    recall_score_val = recall_score(y_val, y_val_pred, average='weighted')
    f1_score_val = f1_score(y_val, y_val_pred, average='weighted')

    # Test metrics
    accuracy_score_test = accuracy_score(y_test, y_test_pred)
    precision_score_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_score_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_score_test = f1_score(y_test, y_test_pred, average='weighted')

    # Specificity and FNR for validation and test sets
    specificity_val, fnr_val = compute_specificity_fnr(y_val, val_probs, 3)  # For validation
    specificity_test, fnr_test = compute_specificity_fnr(y_test, test_probs, 3)  # For test

    # ROC curve and AUC for validation and test
    plot_roc_curve(y_val, val_probs, dataset_type="Validation", save_path="ROC_fig/roc_curve_val.png")
    plot_roc_curve(y_test, test_probs, dataset_type="Test", save_path="ROC_fig/roc_curve_test.png")

    # DCA curve for validation and test
    plot_dca_curve(y_val, val_probs, dataset_type="Validation", save_path="DCA_fig/dca_curve_val.png")  # For validation
    plot_dca_curve(y_test, test_probs, dataset_type="Test", save_path="DCA_fig/dca_curve_test.png")  # For test

    # Print overall metrics for validation and test
    print("Validation Metrics:")
    print(f"Accuracy: {accuracy_score_val:.4f}")
    print(f"Precision: {precision_score_val:.4f}")
    print(f"Recall: {recall_score_val:.4f}")
    print(f"F1 Score: {f1_score_val:.4f}")
    print(f"Specificity: {specificity_val:.4f}")
    print(f"FNR: {fnr_val:.4f}")

    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy_score_test:.4f}")
    print(f"Precision: {precision_score_test:.4f}")
    print(f"Recall: {recall_score_test:.4f}")
    print(f"F1 Score: {f1_score_test:.4f}")
    print(f"Specificity: {specificity_test:.4f}")
    print(f"FNR: {fnr_test:.4f}")

    test_index, test_excel_feature, test_label = extract_excel_features(
        '/tmp/pycharm_project_600/PUMC_LF/编码后汇总-协和+隆福-HER2表达分类-split_data71515-test-del.xlsx')
    # Print the prediction results and probabilities for each patient in the test set
    print("\nTest Set Prediction Results and Probabilities:")
    for i in range(len(y_test_pred)):
        patient_id = test_index[i]  # Assuming you have a unique identifier for each patient, adjust as necessary
        predicted_class = y_test_pred[i]
        predicted_probabilities = test_probs[i]

        print(f"Patient {patient_id}: Predicted Class = {predicted_class}, Probabilities = {predicted_probabilities}")



if __name__ == "__main__":
    SEED = 45
    set_seed(SEED)
    main()
