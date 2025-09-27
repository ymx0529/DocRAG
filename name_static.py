import os
import csv

# path = "/home/ningxy/Experiment/Dataset/TAT-DQA/cache"
path = "/home/hdd/MRAG/Dataset/MMLongBench-Doc/cache"
folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
# 按字母顺序排序（保证每次结果一致）
folders = sorted(folders)

for i in range(len(folders)):
    pdf_name = folders[i]
    print(pdf_name)

with open("MMLongBench-Doc.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["folder_name"])  # 写表头
    for folder in folders:
        writer.writerow([folder])

print("保存完成: MMLongBench-Doc.csv")