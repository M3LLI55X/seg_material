import os
import numpy as np
import cv2
import re

def merge_masks_in_folder(folder_path, start_label=1):
    file_names = os.listdir(folder_path)
    pattern = re.compile(r'(\d+)_mask\.png')
    mask_files = [f for f in file_names if pattern.match(f)]
    mask_files.sort(key=lambda x: int(pattern.match(x).group(1)))
    
    if not mask_files:
        print(f"未在文件夹 {folder_path} 中找到符合命名规则的掩码文件。")
        return None
    
    first_mask_path = os.path.join(folder_path, mask_files[0])
    first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        print(f"无法读取文件：{first_mask_path}")
        return None
    height, width = first_mask.shape
    
    merged_mask = np.zeros((height, width), dtype=np.uint8)
    
    current_label = start_label
    for file_name in mask_files:
        mask_path = os.path.join(folder_path, file_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取文件：{mask_path}")
            continue
        
        _, mask_bin = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
        merged_mask[(mask_bin == 1) & (merged_mask == 0)] = current_label
        
        current_label += 1
    
    return merged_mask

base_folder = '/dtu/blackhole/11/180913/seg_material/experiments/tea'
output_folder = '/dtu/blackhole/11/180913/seg_material/merged_tea'
os.makedirs(output_folder, exist_ok=True)

for i in range(1, 37):
    experiment_folder = os.path.join(base_folder, f'teapot{i}', 'clean_masks', '0')
    merged_mask = merge_masks_in_folder(experiment_folder)
    
    if merged_mask is None:
        continue
    
    experiment_output_path = os.path.join(output_folder, f'teapot{i}.png')
    cv2.imwrite(experiment_output_path, merged_mask)
    print(f"实验 {i} 的合并掩码已保存到：{experiment_output_path}")
