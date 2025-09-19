import os
import json

total_pages = 0
doc_files = [f for f in os.listdir("/home/ningxy/Dataset/OCRresult&QA/TAT-DQA") if f.lower().endswith(".json")]
print(f"\n TAT-DQA文档总数: {len(doc_files)}")
# 读取 JSON 文件
with open("/home/ningxy/Dataset/OCRresult&QA/tatdqa_dataset_test_gold.json", "r", encoding="utf-8") as f:
    TAT_data = json.load(f)
# 统计所有 questions 的数量
total_questions = sum(len(doc.get("questions", [])) for doc in TAT_data)
print("TAT-DQA questions 数量:", total_questions)



total_pages2 = 0
doc_files2 = [f for f in os.listdir("/home/ningxy/Dataset/OCRresult&QA/SlideVQA") if f.lower().endswith(".json")]
print(f"\n SlideVQA文档总数: {len(doc_files2)}")
# 读取 JSON 文件
with open("/home/ningxy/Dataset/OCRresult&QA/SlideVQA_QA.json", "r", encoding="utf-8") as f:
    SlideVQA_data = json.load(f)
# 统计所有 questions 的数量
total_questions = len(SlideVQA_data)
print("TAT-DQA questions 数量:", total_questions)



total_pages3 = 0
doc_files3 = [f for f in os.listdir("/home/ningxy/Dataset/OCRresult&QA/MMLongBench-Doc") if f.lower().endswith(".json")]
print(f"\n MMLongBench文档总数: {len(doc_files3)}")
# 读取 JSON 文件
with open("/home/ningxy/Dataset/OCRresult&QA/MMLongBench-Doc_samples.json", "r", encoding="utf-8") as f:
    MMLongBench_data = json.load(f)
# 统计所有 questions 的数量
total_questions = len(MMLongBench_data)
print("MMLongBench questions 数量:", total_questions)
