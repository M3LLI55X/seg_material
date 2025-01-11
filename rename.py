import os

def rename_files(directory):
    for i in range(36):
        old_name = os.path.join(directory, f"{i}.png")
        new_name = os.path.join(directory, f"image_0_{i}.png")
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Renamed: {old_name} to {new_name}")
        else:
            print(f"File not found: {old_name}")

# 使用你的文件夹路径
directory = "/dtu/blackhole/11/180913/seg_material/data/teapot"
rename_files(directory)