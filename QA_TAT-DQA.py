import os
import sys
import json
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from LLM.LLMclient import ChatModel
from Database.joint import JointHandler
from Database.nebula import NebulaHandler
from Retrieval.ChunkRetriever import get_related_entities
from Retrieval.ChunkRetriever import chunk_loader
from Retrieval.RelevanceScore import chunk_score
from Retrieval.SystemParameter import get_system_parameter, extract_weights
from Generation.Generator import multimodal_generator
from Generation.LLMJudge import judge_answer, score_answer
from Logs.LoggerUtil import get_logger

############################################# 系统配置 ###############################################
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 配置日志
log_dir = Path(__file__).parent / "Logs"    # 创建日志目录：当前文件同级的 logs 文件夹
logger = get_logger(log_dir=log_dir, backup_count=10)
logging.info("\n------ 系统启动 ------\n")

########################################### 大模型API配置 ############################################
"""阿里云 DashScope QWen 模型配置"""
max_token_count = 32000
model="qwen-plus" 
tokenizer = "Qwen/Qwen3-4B-Instruct-2507"
reasoning_model = False
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
vl_model = "qwen-vl-plus"
api_key="sk-8d2b5880bc254f7abfacbb08f0737a92"
# sk-5e1abc2217e5450faaf791d260ed6074
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

##################################### 大模型client及数据库初始化 ######################################
# 初始化 LLM 模型
chatLLM = ChatModel(model=model,
                    reasoning_model=reasoning_model, 
                    api_key=api_key, 
                    base_url=base_url, 
                    temperature=1.0)
chatVLM = ChatModel(model=vl_model, 
                    reasoning_model=False, 
                    api_key=api_key, 
                    base_url=base_url,
                    temperature=1.0)
encoder = SentenceTransformer(embedding_model)

# 初始化向量数据库和图数据库工具
SPACE = "mrag_test"
joint = JointHandler(space_name=SPACE, 
                     nebula_host="127.0.0.1", nebula_port=9669, 
                     milvus_host="127.0.0.1", milvus_port="19530", 
                     collection_name="mrag_test")
joint.setup(embedding_dim=encoder.get_sentence_embedding_dimension())

############################################## 路径设置 ###############################################
# 选择数据集
dataset_name = "TAT-DQA"

# OCR 结果输出路径
ocr_dir = os.path.join(f"/home/hdd/MRAG/Dataset/{dataset_name}")
# RAG 结果输出目录
output_dir = os.path.join(f"Generation/Output")
output_dataset_dir = os.path.join(f"Generation/Output/{dataset_name}")
os.makedirs(output_dataset_dir, exist_ok=True)
# RAG 评价结果输出目录
rag_dir = os.path.join("Generation/Output", f"{dataset_name}_rag.json")
os.makedirs(os.path.dirname(rag_dir), exist_ok=True)
judge_dir = os.path.join("Generation/Output", f"{dataset_name}_judge.csv")

######################################################################################################
# 指定 QA json 文件路径。不同的数据集，需要指定不同的 json 文件路径和不同的处理方式。
json_path = '/home/hdd/MRAG/Dataset/tatdqa_dataset_test_gold.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
# 提取所有 question 和 answer
doc_qa_dict = {}
for doc_questions in data:
    doc = doc_questions['doc']
    pdf_name = doc['uid']    # uid对应PDF的名称
    questions = doc_questions['questions']
    if pdf_name not in doc_qa_dict:
        doc_qa_dict[pdf_name] = []
    for qa in questions:
         doc_qa_dict[pdf_name].append({
             "question": qa['question'], 
             "answer": qa['answer']
             })
# 统计所有 question-answer 的总数
total_qa_count = sum(len(v) for v in doc_qa_dict.values())
print(f"doc_qa_dict 中总共有 {total_qa_count} 个 question-answer 对")

####################################################### 主流程 ########################################
total_count = 0       # 总计数
correct_count = 0     # 正确数
score_sum = 0.0       # 总分数
# 初始化结果列表
qa_data_list = []
# 初始化结果统计 DataFrame
results_df = pd.DataFrame(columns=["question", "answer", "response", "judgment", "score"])

for pdf_name, qa_list in doc_qa_dict.items():

    # pdf_name = "4cbc46fc4b5a4a86cbe15ef28007d948"
    # qa_list = doc_qa_dict[pdf_name]

    json_path = f"./GraphCache/{dataset_name}/FinalGraph/{pdf_name}_final_graph.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        final_graph = json.load(f)
    print(f"--- 知识图谱载入，共包含 {len(final_graph)} 个元素 ---")

    json_path = f"./GraphCache/{dataset_name}/FinalGraph/{pdf_name}_final_graph_with_vector.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        final_graph_with_vectors = json.load(f)
    print(f"当前文档：{pdf_name} ---")
    print(f"--- 知识图谱载入，共包含 {len(final_graph_with_vectors)} 个元素 ---")

    # 实体向量写入向量数据
    joint.ingest_vector_data(final_graph_with_vectors)

    # 写入图数据
    # 批量遍历并处理description字段的换行符
    for item in final_graph_with_vectors:
        if 'description' in item:
            item['description'] = item['description'].replace('\n', ' ')
    # 使用批量写入
    joint.ingest_graph_data_bulk(
        final_graph_with_vectors,
        entity_batch=500,    # 每批最多 500 个实体
        relation_batch=800,  # 每批最多 800 条关系
        reset=True          # 如果 True 会清空并重建 space
    )

    node_count = joint.nebula.get_node_count()  # 获取节点数量
    edge_count = joint.nebula.get_edge_count()  # 获取边数量
    print(f"[Joint] 数据库中当前有 {node_count} 个节点和 {edge_count} 条边")

    # 计算全图的 PageRank 和 Closeness 并导出到 CSV 文件
    nebula_handler = NebulaHandler(space_name=SPACE, host="127.0.0.1", port=9669, user="root", password="nebula")
    try:
        # 用边权：use_edge_weight=True（前提是你的边有 relationship_strength 字段）
        pagerank, closeness = nebula_handler.compute_and_export_algorithms(
            pagerank_csv=os.path.join(output_dir, dataset_name, pdf_name+"_pagerank.csv"),
            closeness_csv=os.path.join(output_dir, dataset_name, pdf_name+"_closeness.csv"),
            use_edge_weight=True,
            closeness_undirected=True
        )
    finally:
        nebula_handler.close()

    # 选择一个问题
    for qa in qa_list:
        question = qa['question']
        answer = qa['answer']
        '''
        根据问题检索
        '''
        # 问题为query
        query = question
        q_vector = encoder.encode(query, convert_to_tensor=False)  # 返回 numpy.ndarray

        # 查询与问题最相似的 entityID
        search_results = joint.search_neighbors_by_vector(q_vector.tolist(), top_k=5, partition=None)  # partition 可设置为 type 名称
        # 获取与问题最相似的实体
        results_matched = get_related_entities(entity_id=search_results, graph_data=final_graph)

        # 调用 LLM 分析问题，并建议权重参数
        rag_parameters = get_system_parameter(model=model, 
                                              reasoning_model=reasoning_model, 
                                              query=query, 
                                              api_key=api_key, 
                                              base_url=base_url)
        # 从 LLM 响应中提取权重参数
        parameters = extract_weights(rag_parameters)
        alpha = parameters.get("alpha")
        beta = parameters.get("beta")
        lam = parameters.get("lam")

        # 在子图中，计算每个 chunk 的排名 score(c)
        scores_text, scores_multimodal = chunk_score(query=query, 
                                                     graph_data=results_matched, 
                                                     pagerank=pagerank, 
                                                     closeness=closeness, 
                                                     encoder_model=encoder, 
                                                     alpha=alpha, 
                                                     beta=beta, 
                                                     lam=lam)

        # 根据 score，选出 top_k 个 chunk
        chunks = chunk_loader(ocr_json_path=Path(ocr_dir), 
                              pdf_name=pdf_name, 
                              scores_text=scores_text, 
                              scores_multimodal=scores_multimodal, 
                              top_k_text=5, 
                              top_k_multimodal=3)
        '''
        生成回答以及提取关键信息
        '''
        # 将top_k个chunk的内容进行处理，提取image和text，连同query，一起输入视觉大模型，生成回答
        ocr_imagefile_dir = os.path.join(ocr_dir, "cache", pdf_name, "auto")
        prompt, image_list, response = multimodal_generator(query=query, 
                                                            answer=answer, 
                                                            chunks=chunks, 
                                                            chatVLM=chatVLM, 
                                                            ocr_imagefile_dir=Path(ocr_imagefile_dir), 
                                                            save_dir=output_dataset_dir)
        qa_data = {"query": query, 
                   "prompt": prompt, 
                   "image": image_list, 
                   "response": response, 
                   "answer": answer}
        qa_data_list.append(qa_data)

        # 回答正确性判断
        judgment = judge_answer(model=model, 
                                reasoning_model=reasoning_model, 
                                question=question, 
                                reference_answer=answer, 
                                response=response, 
                                api_key=api_key, 
                                base_url=base_url)
        print(judgment)
        if judgment.lower() in ["true", "false"]:
            if judgment.lower() == "true":
                correct_count += 1
        total_count += 1
        # 回答得分计算
        score = score_answer(model=model, 
                             reasoning_model=reasoning_model, 
                             question=question, 
                             reference_answer=answer, 
                             response=response, 
                             api_key=api_key, 
                             base_url=base_url)
        print(score)
        try:
            score_value = float(score)
        except ValueError:
            score_value = 0.0
        score_sum += score_value

        results_df.loc[len(results_df)] = [question, answer, response, judgment, score]

# 计算准确率和平均得分
accuracy = correct_count / total_count
average_score = score_sum / total_count
logging.info(f"total_count: {total_count}")
logging.info(f"correct_count: {correct_count}, Accuracy: {accuracy}")
logging.info(f"score_sum: {score_sum}, Average Score: {average_score}")

# 保存结果到 CSV 文件
results_df.to_csv(judge_dir, index=False, encoding="utf-8-sig")
# 循环结束后，将所有 data 写入 JSON 文件
with open(rag_dir, "w", encoding="utf-8") as f:
    json.dump(qa_data_list, f, ensure_ascii=False, indent=2)

logging.info("\n------ 任务结束 ------\n")