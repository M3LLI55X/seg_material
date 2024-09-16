import numpy as np
from PIL import Image
import os

def load_all_mask_images(mask_folder):
    mask_arrays = {}
    for root, _, files in os.walk(mask_folder):
        for file in files:
            if file.endswith('_mask.png'):
                k = int(file.split('_')[0])  # 提取k值
                mask_path = os.path.join(root, file)
                mask_image = Image.open(mask_path)
                mask_array = np.array(mask_image)
                mask_arrays[k] = mask_array
    return mask_arrays

def create_large_matrix(mask_arrays):
    # 假设所有mask图像的尺寸相同
    sample_mask = next(iter(mask_arrays.values()))
    height, width = sample_mask.shape
    large_matrix = np.zeros((height, width), dtype=int)
    return large_matrix

def merge_masks_into_large_matrix(mask_arrays, large_matrix):
    for k, mask_array in mask_arrays.items():
        large_matrix[mask_array > 0] = k  # 将mask中非零位置的值设为k
    return large_matrix

def create_color_mapping(matrix):
    unique_values = np.unique(matrix)
    color_mapping = {}
    np.random.seed(0)  # 固定随机种子以确保颜色一致
    for value in unique_values:
        color_mapping[value] = tuple(np.random.randint(0, 256, size=3))
    return color_mapping

def generate_color_image(matrix, color_mapping):
    height, width = matrix.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in color_mapping.items():
        color_image[matrix == value] = color
    return color_image

def save_color_image(color_image, output_path):
    image = Image.fromarray(color_image)
    image.save(output_path)

if __name__ == '__main__':
    # 加载所有mask图像
    mask_folder = '/dtu/blackhole/11/180913/Make_it_Real/experiments/chair/clean_masks/0'
    mask_arrays = load_all_mask_images(mask_folder)
    
    # 创建一个大的矩阵
    large_matrix = create_large_matrix(mask_arrays)
    
    # 合并mask图像
    large_matrix = merge_masks_into_large_matrix(mask_arrays, large_matrix)
    
    # 创建颜色映射
    color_mapping = create_color_mapping(large_matrix)
    
    # 生成彩色图像
    color_image = generate_color_image(large_matrix, color_mapping)
    
    # 保存彩色图像
    output_path = f'/experiments/chair/output_image.png'
    save_color_image(color_image, output_path)