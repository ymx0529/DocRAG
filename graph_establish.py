import os
import sys
import json
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from LLM.LLMclient import ChatModel
from DocProcess.DocumentProcessor import parsed_document_process
from GraphProcess.EntityDisambiguation import run_disambiguation_pipeline
from GraphProcess.GraphVectorProcessor import process_vectors_and_clean_graph
from Logs.LoggerUtil import get_logger

############################################# 系统配置 ###############################################
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 配置日志
log_dir = Path(__file__).parent / "Logs"    # 创建日志目录：当前文件同级的 logs 文件夹
logger = get_logger(log_dir=log_dir, backup_count=10)
logging.info("\n------ 系统启动 ------\n")

########################################### 大模型API配置 ############################################
"""本地 Ollama 模型配置"""
# max_token_count = 32768
# model="qwen3:30b-a3b-instruct-2507-fp16"            # ollama 模型名称
# tokenizer = "Qwen/Qwen3-30B-A3B-Instruct-2507"      # huggingface 模型名称, 用于计算 token 数, 要和 model 匹配
# reasoning_model = False                             # 是否是推理模型，有无</think>符号
# embedding_model = "Qwen/Qwen3-Embedding-0.6B"       # 用于获取语义向量，做相似度检索，无需与 LLM 匹配
# vl_model="qwen2.5vl:32b"                            # 多模态大模型
# api_key="ollama"                                    # 本地 ollama 服务的 api key
# base_url="http://localhost:11434/v1/"               # ollama 服务地址

"""阿里云 DashScope QWen 模型配置"""
max_token_count = 32000
model="qwen-plus" 
tokenizer = "Qwen/Qwen3-4B-Instruct-2507"
reasoning_model = False
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
vl_model = "qwen-vl-plus-2025-08-15"
api_key="sk-ab4c719f273941309717ac10fef2f927"
# sk-5e1abc2217e5450faaf791d260ed6074
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

##################################### 大模型client及数据库初始化 ######################################
# 初始化 LLM 模型
chatLLM = ChatModel(model=model,
                    reasoning_model=reasoning_model, 
                    api_key=api_key, 
                    base_url=base_url)
chatVLM = ChatModel(model=vl_model, 
                    reasoning_model=False, 
                    api_key=api_key, 
                    base_url=base_url)
encoder = SentenceTransformer(embedding_model)

######################################################################################################

# 选择数据集
dataset_name = "TAT-DQA"
# 读取文件名列表
df = pd.read_csv("TAT-DQA_folders.csv")

# 取出前 100 个文件夹名
folder_name = df["folder_name"].iloc[:100].tolist()

for i in range(len(folder_name)):

    # 目标 PDF 文档名称
    pdf_name = folder_name[i]
    print(pdf_name)

    # OCR 结果输出路径
    ocr_dir = os.path.join("Dataset", dataset_name)
    # OCR 图片输出路径
    ocr_imagefile_dir = os.path.join("Dataset", dataset_name, "cache", pdf_name, "auto")

    '''
    文档内容读取与建图
    '''
    graph = parsed_document_process(pdf_name=pdf_name, 
                                    json_file_path=Path(ocr_dir), 
                                    image_file_dir=Path(ocr_imagefile_dir), 
                                    chatLLM=chatLLM, 
                                    chatVLM=chatVLM,
                                    max_token_count=max_token_count, 
                                    model=model, 
                                    encoding_model=tokenizer, 
                                    encoder=encoder, 
                                    json_mode=True)   # 必须使用 JSON 格式的三元组提取
    print(f"--- 知识图谱初步构建完成，共包含 {len(graph)} 个元素 ---")

    # --- 保存原始的、未消歧的图谱 ---
    origin_graph_dir = Path(f"GraphCache/{dataset_name}/OriginalGraph")
    origin_graph_dir.mkdir(parents=True, exist_ok=True)
    origin_graph_path = origin_graph_dir / f"{pdf_name}_original_graph.json"
    with open(origin_graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=4, ensure_ascii=False)
        print(f"最终的知识图谱到保存为{origin_graph_path}")


    # 对生成的图谱进行实体消歧
    final_graph_with_vectors = run_disambiguation_pipeline(graph, similarity_threshold=0.95)
    final_graph = process_vectors_and_clean_graph(final_graph_with_vectors=final_graph_with_vectors, 
                                                #vector_store=joint_handler
                                                # vector_store=None
                                                )

    # --- 保存最终的、消歧后的图谱 ---
    final_graph_dir = Path(f"GraphCache/{dataset_name}/FinalGraph")
    final_graph_dir.mkdir(parents=True, exist_ok=True)
    final_graph_with_vector_path = final_graph_dir / f"{pdf_name}_final_graph_with_vector.json"
    with open(final_graph_with_vector_path, 'w', encoding='utf-8') as f:
        json.dump(final_graph_with_vectors, f, indent=4, ensure_ascii=False)
        print(f"最终的知识图谱到保存为{final_graph_with_vector_path}")
    final_graph_path = final_graph_dir / f"{pdf_name}_final_graph.json"
    with open(final_graph_path, 'w', encoding='utf-8') as f:
        json.dump(final_graph, f, indent=4, ensure_ascii=False)
        print(f"最终的知识图谱到保存为{final_graph_path}")

