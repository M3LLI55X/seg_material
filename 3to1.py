import os
import cv2

# 指定输入和输出文件夹路径
input_folder = '/dtu/blackhole/11/180913/seg_material/merged_results'  # 替换为包含36张图像的文件夹路径
output_folder = '/dtu/blackhole/11/180913/seg_material/merged_result'  # 替换为保存单通道图像的文件夹路径

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 遍历每张图像，将其转换为单通道并保存
for image_file in image_files[:36]:
    # 读取三通道图像
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    
    # 提取图像的R通道
    r_channel_image = image[:, :, 2]
    
    # 保存单通道图像
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, r_channel_image)

print("所有图像已成功转换为单通道并保存。")
