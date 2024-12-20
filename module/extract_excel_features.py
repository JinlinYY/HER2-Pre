import numpy as np
import pandas as pd

# def extract_excel_features(filename):
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#     nplist = readbook.to_numpy()
#     index = nplist[:, 0]  # 获取序号列
#     feature = nplist[:, 1:-1]  # 获取特征列，排除第一列序号和最后一列标签
#     feature = np.float64(feature)
#     target = nplist[:, -1]  # 获取标签列
#     return index, feature, target

def extract_excel_features(filename):
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')
    index = readbook.iloc[:, 0].to_numpy()
    labels = readbook.iloc[:, -1].to_numpy()
    features_df = readbook.iloc[:, 1:-1]
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)

    return index, combined_features, labels

# def extract_excel_features(filename):
#     """
#     提取 Excel 文件中的特征和标签。
#
#     Args:
#         filename (str): 不带扩展名的文件路径。
#
#     Returns:
#         tuple: index (序号列), feature (特征数组), target (标签列)。
#     """
#     # 读取 Excel 文件
#     readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
#     nplist = readbook.to_numpy()
#
#     # 获取序号列
#     index = nplist[:, 0]
#
#     # 获取特征列，排除序号和标签列
#     feature = nplist[:, 1:-1]
#
#     # 转换特征列为浮点数，同时处理非数值数据
#     try:
#         feature = np.array(feature, dtype=np.float64)
#     except ValueError:
#         print("Warning: Non-numeric data found in feature columns. Attempting to clean data.")
#         # 将非数值替换为 NaN
#         feature = pd.DataFrame(feature).apply(pd.to_numeric, errors='coerce').to_numpy()
#
#         # 检查并提示用户非数值数据位置
#         if np.isnan(feature).any():
#             nan_indices = np.argwhere(np.isnan(feature))
#             print("Non-numeric data detected at the following positions:")
#             for row, col in nan_indices:
#                 print(f"Row {row + 1}, Column {col + 2}")  # Adjust indices to match Excel (1-based indexing)
#
#             # 填充 NaN 或根据需求删除
#             feature = np.nan_to_num(feature)  # 用0替换NaN，或者根据实际需求修改处理策略
#
#     # 获取标签列
#     target = nplist[:, -1]
#
#     return index, feature, target