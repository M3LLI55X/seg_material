import cv2
import numpy as np

def open_mask_as_array(mask_path):
    # 读取掩码图片，使用灰度模式
    mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_array is None:
        print(f"无法读取文件：{mask_path}")
    return mask_array

def count_unique_elements(matrix):
    # 使用 np.unique 获取矩阵中所有唯一元素及其计数
    unique_elements, counts = np.unique(matrix, return_counts=True)
    return unique_elements, counts

# 示例用法
mask_path = '/dtu/blackhole/11/180913/seg_material/merged_result/chair1.png'
mask_array = open_mask_as_array(mask_path)
if mask_array is not None:
    unique_elements, counts = count_unique_elements(mask_array)

    print(f"矩阵中有 {len(unique_elements)} 种不同的元素：")
    for element, count in zip(unique_elements, counts):
        print(f"元素 {element} 出现了 {count} 次")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(mask[0])