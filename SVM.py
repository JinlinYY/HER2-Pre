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
from sklearn.svm import SVC
def main():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    # Load data
    index, excel_feature, label = extract_excel_features('HER2_excel_data/HER2-data-特征取交集后汇总')
    excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # Extract image features
    image_filenames = ['HER2_image_data-roi-1/{}.jpg'.format(idx) for idx in index.astype(int)]
    image_features = extract_image_features(image_filenames)
    pca = PCA(n_components=35)
    image_features_pca = pca.fit_transform(image_features)
    image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)

    # Combine features
    combined_features = combine_features(image_features_pca_tensor, excel_feature_tensor)
    combined_features_tensor, label_tensor = inputtotensor(combined_features, label)

    # K-fold cross-validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    all_metrics = {"Validation": [], "Test": []}
    accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro = [], [], [], [], [], []
    all_y_true, all_y_probs = [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features, label):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features[train_index], combined_features[test_index]
        y_train, y_test = label[train_index], label[test_index]

        # Initialize SVM classifier
        svm_clf = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=SEED)

        # Train the SVM
        svm_clf.fit(x_train, y_train)

        # Predictions
        y_val_pred = svm_clf.predict(x_test)
        y_test_pred = y_val_pred  # No separate validation set in this context, same as test
        val_probs = svm_clf.predict_proba(x_test)
        test_probs = val_probs

        # Metrics calculation
        cm_val = confusion_matrix(y_test, y_val_pred)
        cm_test = cm_val  # Since we're using the same dataset
        validation_metrics = calculate_metrics(cm_val, dataset_type="Validation")
        test_metrics = calculate_metrics(cm_test, dataset_type="Test")
        all_metrics["Validation"].append(validation_metrics)
        all_metrics["Test"].append(test_metrics)

        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_test_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_test_pred, average='weighted'))

        # ROC curve and AUC for the current fold
        all_y_true.extend(y_test)
        all_y_probs.extend(test_probs[:, 1])  # Assuming binary classification
        roc_auc_fold = plot_roc_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test")
        AUC_score_macro.append(roc_auc_fold['macro'])
        AUC_score_micro.append(roc_auc_fold['micro'])

        # Visualize predictions
        visualize_predictions(y_test, y_test_pred, dataset_type=f"Fold {fold} Test")

    print_mean_std_metrics(all_metrics)
    print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro)

    # Compute overall ROC curve and AUC value
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    overall_roc_auc = plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall")
    print(overall_roc_auc)


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    main()
