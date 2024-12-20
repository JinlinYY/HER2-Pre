import os
import shutil
import pandas as pd
import numpy as np

def open_excel(filename):
    """
    打开数据集，进行数据处理
    :param filename: 文件名
    :return: 序号、特征集数据、标签集数据
    """
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')
    nplist = readbook.to_numpy()
    index = nplist[:, 0]  # 获取序号列
    # feature = nplist[:, 1:-1]  # 获取特征列，排除第一列序号和最后一列标签
    # feature = nplist[:, 1:12]  # 获取特征列，排除第一列序号和最后一列标签
    # feature = np.float64(feature)
    # target = nplist[:, -1]  # 获取标签列
    return index#, feature, target

def copy_first_image_for_idx(idx, source_dir, target_dir):
    """
    根据 idx 复制对应文件夹下以“idx-”开头的第一张.jpg图片到目标文件夹
    :param idx: 待查找的 idx
    :param source_dir: 源文件夹路径，包含以“idx-”开头的图片
    :param target_dir: 目标文件夹路径，用于存放复制的图片
    """
    # 构建以 idx- 开头的文件名
    prefix = '{}-'.format(idx)
    found_matching_image = False
    
    # 遍历源文件夹中以 prefix 开头的文件
    for filename in os.listdir(source_dir):
        if filename.startswith(prefix) and filename.endswith('.jpg'):
            # 找到以 prefix 开头且是 .jpg 格式的文件，复制到目标文件夹
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            
            # 复制文件（仅复制第一张符合条件的图片）
            if os.path.isfile(source_file):
                shutil.copyfile(source_file, target_file)
                found_matching_image = True
                break  # 仅复制第一张符合条件的图片
    
    # 如果未找到符合条件的图片，则打印出 prefix
    if not found_matching_image:
        print("未找到以 '{}' 开头且是 .jpg 格式的图片".format(prefix))

def process_data_folder(index, data_dir, target_dir):
    """
    处理数据文件夹中的图片，根据文件名以及 idx 进行复制
    :param data_dir: 数据文件夹路径，包含以“idx-”开头的图片
    :param target_dir: 目标文件夹路径，用于存放复制的图片
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for idx in index:
        copy_first_image_for_idx(int(idx), data_dir, target_dir)

index = open_excel('/tmp/pycharm_project_600/PUMC_LF/编码后汇总-协和+隆福-HER2表达分类.xlsx')

#改名 只保留序号
def rename_jpg_files(directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.bmp'):
            # 获取文件名中“-”前的部分
            new_filename = filename.split('-')[0] + '.bmp'
            # 构建新的文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_path, new_path)
            # print(f"重命名文件: {old_path} -> {new_path}")

# 示例用法
data_folder = '/tmp/pycharm_project_600/PUMC_LF/US'  # 数据文件夹路径，包含以“idx-”开头的图片
target_folder = '/tmp/pycharm_project_600/PUMC_LF/US_data'  # 目标文件夹路径，用于存放复制的图片

# 处理数据文件夹中的图片
process_data_folder(index, data_folder, target_folder)
print("图片复制完成！")

# 调用函数并指定目录
rename_jpg_files(target_folder)
print("图片改名完成！")
