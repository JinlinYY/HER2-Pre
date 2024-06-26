import numpy as np
import pandas as pd

def extract_excel_features(filename):
    readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
    nplist = readbook.to_numpy()
    index = nplist[:, 0]  # 获取序号列
    feature = nplist[:, 1:-1]  # 获取特征列，排除第一列序号和最后一列标签
    feature = np.float64(feature)
    target = nplist[:, -1]  # 获取标签列
    return index, feature, target

