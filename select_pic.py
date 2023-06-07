import os
import random
import shutil

# 指定原始图片路径和随机选择的图片路径
original_dir = "/path/to/original/images"
random_dir = "/path/to/random/images"

# 获取原始图片列表
original_images = os.listdir(original_dir)

# 计算应该选择的图片数量
num_images_to_select = int(len(original_images) * 0.5)

# 随机选择图片
selected_images = random.sample(original_images, num_images_to_select)

# 将选中的图片复制到随机选择的图片路径中
if not os.path.exists(random_dir):
    os.makedirs(random_dir)
for image in selected_images:
    src_path = os.path.join(original_dir, image)
    dst_path = os.path.join(random_dir, image)
    shutil.copy(src_path, dst_path)

# # 删除未被选中的图片
# for image in original_images:
#     if image not in selected_images:
#         os.remove(os.path.join(original_dir, image))