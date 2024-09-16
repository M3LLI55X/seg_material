from utils.gen_config import *
import sys 
sys.path.extend(['./som', './scripts/kaolin_scripts'])
from som.sesam_serial import seg_serial
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


if __name__ == '__main__':
    # 加载mask图像
    mask_path = 'path_to_mask_image.png'
    mask_array = load_mask_image(mask_path)
    
    # 加载JSON结果
    json_path = 'path_to_json_result.json'
    json_data = load_json_result(json_path)
    
    # 创建材质映射
    material_mapping = create_material_mapping(json_data)
    
    # 生成二维数组
    material_array = generate_material_array(mask_array, material_mapping)
    
    # 打印或保存结果
    print(material_array)

    # 查看图片文件的矩阵大小和通道数
    image_path = 'path_to_image_file.png'
    image = Image.open(image_path)
    image_array = np.array(image)
    print(f"Image shape: {image_array.shape}")