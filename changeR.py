import os
import shutil

# 指定原始目录路径和目标目录路径
source_directory = "./dataset/test/tir"
target_directory = "./dataset/test/ir"

# 确保目标目录存在，如果不存在则创建它
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 列出原始目录中的所有文件
files = os.listdir(source_directory)

# 循环遍历文件
for filename in files:
    # 检查文件名是否以数字+R.jpg结尾
    if filename.endswith("R.jpg"):
        # 构建新文件名，去掉最后一个字符（"R"）并保留文件扩展名
        new_filename = filename[:-5] + ".jpg"
        # 构建原始文件路径和目标文件路径
        source_path = os.path.join(source_directory, filename)
        target_path = os.path.join(target_directory, new_filename)
        # 复制文件到目标目录
        shutil.copy(source_path, target_path)
