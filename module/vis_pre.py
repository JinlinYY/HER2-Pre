import matplotlib.pyplot as plt

def visualize_predictions(y_true, y_pred, dataset_type):
    plt.figure(figsize=(20, 3))
    plt.scatter(range(len(y_true)), y_true, label='True Label', marker='o', color='b', s=100, alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted Label', marker='^', color='r', s=100, alpha=0.7)
    plt.title(f'{dataset_type} - True and Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks([0, 1, 2], ["HER2-low", "HER2-zero", "HER2-positive"])  # 设置纵坐标刻度及标签
    plt.legend()
    plt.grid(True)
    plt.show()