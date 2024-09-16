from utils.merge import Merge
from som.seg import test_single_image
from utils.gen_config import *
import sys 
sys.path.extend(['./som', './scripts/kaolin_scripts'])
# from som.sesam_serial import seg_serial
import subprocess

from utils.gpt4_query import query_overall, query_refine, query_description, query_shape
from utils.mask_postprocess import refine_masks
from utils.texture_postprocess import region_refine, pixel_estimate
from utils.tools import *

from scripts.kaolin_scripts.load_cfg import render_model, paint_model_w_mask

import os
import argparse
import warnings

from PIL import Image
import numpy as np
import json

def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp_name", default='chair', type=str, help="Experiment name(unique)")
    parser.add_argument("--fine_grain", default=False, type=bool, help="Segmentation grain.")
    parser.add_argument("--view_num", default=1, type=int, help="Number of view points.")
    parser.add_argument("--img_dir", default='data/images', type=str, help="Directory of image file.")
    argv = parser.parse_args()
    return argv

def load_mask_image(mask_path):
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)
    return mask_array

def load_json_result(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def create_material_mapping(json_data):
    material_mapping = {}
    for key, value in json_data.items():
        material_mapping[int(key)] = value
    return material_mapping

def generate_material_array(mask_array, material_mapping):
    material_array = np.zeros_like(mask_array, dtype=object)
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            material_id = mask_array[i, j]
            material_array[i, j] = material_mapping.get(material_id, 'Unknown')
    return material_array



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
    argv = args()
    exp_name = argv.exp_name  
    os.makedirs(f'/dtu/blackhole/11/180913/Make_it_Real/experiments/{exp_name}', exist_ok=True)
    folder_path = f'/dtu/blackhole/11/180913/Make_it_Real/experiments/{exp_name}'
    result_path = f'/dtu/blackhole/11/180913/Make_it_Real/experiments/{exp_name}'

    test_image_path=argv.img_dir
    test_single_image(test_image_path, exp_name)
    print("test_single_image called successfully")
    Merge(test_image_path,exp_name)
    print("Merge called successfully")
    
    # OpenAI API Key
    api_key = 'store in doc'
    
    leave_index = get_max_area(result_path)
    if argv.view_num > 1:
        if len(argv.leave_list) > 0: leave_list = argv.leave_list
        else:
            if leave_index < 8: leave_list = [leave_index, 7-leave_index]
            else: leave_list = [leave_index, 0]
    else:
        leave_list = [leave_index]

    view_path = f'/data/images'
    mv_img_pth = paste_image(view_path)
    obj_info = query_shape(folder_path, mv_img_pth, api_key)

    # GPT-4V query about material of each part
    query_overall(folder_path, leave_list, api_key, obj_info)
    query_refine(folder_path, leave_list, api_key, obj_info)
    cache_path = query_description(folder_path, leave_list, api_key, obj_info, force=True)

    # sort gpt result
    most_result_pth, _ = sort_gpt_result(cache_path, leave_list)
    mat2im_path = sort_categories(most_result_pth)



    # 加载所有mask图像
    mask_folder = f'/experiments/{exp_name}/clean_masks/0'
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
    output_path = 'output_image.png'
    save_color_image(color_image, output_path)
    