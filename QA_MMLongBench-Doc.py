import os 
import sys
import json
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import re
import time
from typing import Set, List
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


def parse_evidence_pages(evidence_field):
    """
    兼容 evidence_pages 的多种格式：
      - [15, 16]
      - "[19, 20]"
      - "19,20"
      - "19"
    返回 set[int]
    """
    if evidence_field is None:
        return set()
    if isinstance(evidence_field, list):
        try:
            return set(int(x) for x in evidence_field)
        except Exception:
            s = json.dumps(evidence_field)
            nums = re.findall(r'\d+', s)
            return set(int(x) for x in nums)
    if isinstance(evidence_field, str):
        s = evidence_field.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return set(int(x) for x in parsed)
            if isinstance(parsed, int):
                return {parsed}
        except Exception:
            nums = re.findall(r'\d+', s)
            return set(int(x) for x in nums)
    try:
        return {int(evidence_field)}
    except Exception:
        return set()


def predicted_pages_from_chunks(chunks: dict):
    """从 chunks 中提取预测页码集合"""
    pages = set()
    for d in ("text", "multimodal"):
        for key in chunks.get(d, {}).keys():
            if isinstance(key, (list, tuple)) and len(key) >= 1:
                try:
                    pages.add(int(key[0]))
                except Exception:
                    pass
            elif isinstance(key, (int, float)):
                pages.add(int(key))
            elif isinstance(key, str):
                m = re.search(r'\d+', key)
                if m:
                    pages.add(int(m.group()))
    return pages


def prf_from_sets(predicted: set, gold: set):
    """计算 Precision / Recall / F1"""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


############################################# 系统配置 ###############################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = Path(__file__).parent / "Logs/Log1"    
logger = get_logger(log_dir=log_dir, backup_count=10)
logging.info("\n------ 系统启动 ------\n")


########################################### 大模型API配置 ############################################
max_token_count = 32000
model="qwen-plus" 
tokenizer = "Qwen/Qwen3-4B-Instruct-2507"
reasoning_model = False
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
vl_model = "qwen-vl-plus"
api_key="sk-8d2b5880bc254f7abfacbb08f0737a92"   # 请替换
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"


##################################### 大模型client及数据库初始化 ######################################
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

SPACE = "mrag_sub1"
joint = JointHandler(space_name=SPACE, 
                     nebula_host="127.0.0.1", nebula_port=9669, 
                     milvus_host="127.0.0.1", milvus_port="19530", 
                     collection_name="mrag_sub1")
joint.setup(embedding_dim=encoder.get_sentence_embedding_dimension())


############################################## 路径设置 ###############################################
dataset_name = "MMLongBench-Doc"
ocr_dir = os.path.join(f"/home/hdd/MRAG/Dataset/{dataset_name}")
output_dir = os.path.join(f"Generation/Output")
output_dataset_dir = os.path.join(f"Generation/Output/{dataset_name}")
os.makedirs(output_dataset_dir, exist_ok=True)
rag_dir = os.path.join("Generation/Output", f"{dataset_name}_rag.json")
os.makedirs(os.path.dirname(rag_dir), exist_ok=True)
judge_dir = os.path.join("Generation/Output", f"{dataset_name}_judge.csv")

json_path = '/home/hdd/MRAG/Dataset/MMLongBench-Doc_samples.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ======== 修改点1：保留 evidence_pages，并做 -1 修正 =========
doc_qa_dict = {}
for item in data:
    pdf_name = item['doc_id']
    question = item['question']
    answer = item['answer']
    evidence_field = item.get("evidence_pages", None)

    gold_pages = parse_evidence_pages(evidence_field)
    # 减1（因为 gold 是从 1 开始，而 OCR 预测是从 0 开始）
    gold_pages = {p-1 for p in gold_pages if p > 0}

    if pdf_name.endswith(".pdf"):
        pdf_name = pdf_name[:-4] 
    if pdf_name not in doc_qa_dict:
        doc_qa_dict[pdf_name] = []

    doc_qa_dict[pdf_name].append({
        "question": question,
        "answer": answer,
        "gold_pages": gold_pages   # 保存
    })

total_qa_count = sum(len(v) for v in doc_qa_dict.values())
print(f"doc_qa_dict 中总共有 {total_qa_count} 个 question-answer 对")


####################################################### 主流程 ########################################
total_count = 0       
correct_count = 0     
score_sum = 0.0       

qa_data_list = []

results_df = pd.DataFrame(columns=[
    "question", "answer", "response", "judgment", "score",
    "predicted_pages", "gold_pages", "precision", "recall", "f1"
])

for pdf_name, qa_list in doc_qa_dict.items():
    json_path_1 = f"/home/yuanxy/Experiment_yxy/GraphCache/{dataset_name}/FinalGraph/{pdf_name}_final_graph.json"
    json_path_2 = f"/home/yuanxy/Experiment_yxy/GraphCache/{dataset_name}/FinalGraph/{pdf_name}_final_graph_with_vector.json"

    if not (os.path.exists(json_path_1) and os.path.exists(json_path_2)):
        continue

    with open(json_path_1, 'r', encoding='utf-8') as f:
        final_graph = json.load(f)
    with open(json_path_2, 'r', encoding='utf-8') as f:
        final_graph_with_vectors = json.load(f)

    joint.ingest_vector_data(final_graph_with_vectors)
    for item in final_graph_with_vectors:
        if 'description' in item:
            item['description'] = item['description'].replace('\n', ' ')
    joint.ingest_graph_data_bulk(
        final_graph_with_vectors,
        entity_batch=500, relation_batch=800, reset=True
    )

    nebula_handler = NebulaHandler(space_name=SPACE, host="127.0.0.1", port=9669, user="root", password="nebula")
    try:
        pagerank, closeness = nebula_handler.compute_and_export_algorithms(
            pagerank_csv=os.path.join(output_dir, dataset_name, pdf_name+"_pagerank.csv"),
            closeness_csv=os.path.join(output_dir, dataset_name, pdf_name+"_closeness.csv"),
            use_edge_weight=True, closeness_undirected=True
        )
    finally:
        nebula_handler.close()

    for qa in qa_list:
        question = qa['question']
        answer = qa['answer']
        gold_pages = qa.get("gold_pages", set())

        # ======== 修改点2：如果 gold_pages 为空，跳过该 QA =========
        if not gold_pages:
            continue

        # === RAG 检索流程 ===
        query = question
        q_vector = encoder.encode(query, convert_to_tensor=False)
        search_results = joint.search_neighbors_by_vector(q_vector.tolist(), top_k=5, partition=None)
        results_matched = get_related_entities(entity_id=search_results, graph_data=final_graph)

        default_params = {"alpha": 0.4, "beta": 0.3, "lam": 0.3}
        max_retries = 3
        parameters = None

        for attempt in range(max_retries):
            try:
                rag_parameters = get_system_parameter(
                    model=model, reasoning_model=reasoning_model,
                    query=query, api_key=api_key, base_url=base_url
                )
                if isinstance(rag_parameters, list) and rag_parameters:
                    rag_parameters = rag_parameters[0]
                if isinstance(rag_parameters, dict):
                    parameters = extract_weights(rag_parameters)
                    break
            except Exception as e:
                logging.warning(f"[{pdf_name}] 参数提取失败 (第 {attempt+1} 次): {e}")
                time.sleep(1)

        if parameters is None:
            parameters = default_params

        alpha = parameters.get("alpha", default_params["alpha"])
        beta  = parameters.get("beta",  default_params["beta"])
        lam   = parameters.get("lam",   default_params["lam"])

        scores_text, scores_multimodal = chunk_score(
            query=query, graph_data=results_matched,
            pagerank=pagerank, closeness=closeness,
            encoder_model=encoder, alpha=alpha, beta=beta, lam=lam
        )

        chunks = chunk_loader(
            ocr_json_path=Path(ocr_dir),
            pdf_name=pdf_name,
            scores_text=scores_text, scores_multimodal=scores_multimodal,
            top_k_text=7, top_k_multimodal=5
        )

        pred_pages = predicted_pages_from_chunks(chunks)
        precision, recall, f1 = prf_from_sets(pred_pages, gold_pages)

        # === 输出每个问题的 pred_pages 和 gold_pages + 详细计算过程 ===
        print(json.dumps({
            "question": question,
            "pred_pages": sorted(list(pred_pages)),
            "gold_pages": sorted(list(gold_pages))
        }, ensure_ascii=False))
        tp = len(pred_pages & gold_pages)
        fp = len(pred_pages - gold_pages)
        fn = len(gold_pages - pred_pages)
        print(f"计算过程: TP={tp}, FP={fp}, FN={fn} -> Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # === 生成回答 ===
        ocr_imagefile_dir = os.path.join(ocr_dir, "cache", pdf_name, "auto")
        prompt, image_list, response = multimodal_generator(
            query=query, answer=answer, chunks=chunks,
            chatVLM=chatVLM, ocr_imagefile_dir=Path(ocr_imagefile_dir),
            save_dir=output_dataset_dir
        )

        qa_data = {"query": query, "prompt": prompt, "image": image_list,
                   "response": response, "answer": answer}
        qa_data_list.append(qa_data)

        judgment = judge_answer(model=model, reasoning_model=reasoning_model,
                                question=question, reference_answer=answer,
                                response=response, api_key=api_key, base_url=base_url)
        if judgment.lower() == "true":
            correct_count += 1
        total_count += 1

        score = score_answer(model=model, reasoning_model=reasoning_model,
                             question=question, reference_answer=answer,
                             response=response, api_key=api_key, base_url=base_url)
        try:
            score_value = float(score)
        except ValueError:
            score_value = 0.0
        score_sum += score_value

        results_df.loc[len(results_df)] = [
            question, answer, response, judgment, score,
            list(pred_pages), list(gold_pages), precision, recall, f1
        ]


# === 汇总指标 ===
accuracy = correct_count / total_count if total_count > 0 else 0.0
average_score = score_sum / total_count if total_count > 0 else 0.0

total_tp = total_fp = total_fn = 0
for _, row in results_df.iterrows():
    pset = set(row["predicted_pages"])
    gset = set(row["gold_pages"])
    if not gset:
        continue
    total_tp += len(pset & gset)
    total_fp += len(pset - gset)
    total_fn += len(gset - pset)

micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

macro_prec = results_df["precision"].mean() if not results_df.empty else 0.0
macro_rec  = results_df["recall"].mean() if not results_df.empty else 0.0
macro_f1   = results_df["f1"].mean() if not results_df.empty else 0.0

logging.info(f"total_count: {total_count}")
logging.info(f"correct_count: {correct_count}, Accuracy: {accuracy:.4f}")
logging.info(f"score_sum: {score_sum}, Average Score: {average_score:.4f}")
logging.info("页级评估结果：")
logging.info(f"Micro Precision: {micro_prec:.4f}, Micro Recall: {micro_rec:.4f}, Micro F1: {micro_f1:.4f}")
logging.info(f"Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}, Macro F1: {macro_f1:.4f}")
logging.info("\n------ 任务结束 ------\n")

results_df.to_csv(judge_dir, index=False, encoding="utf-8-sig")
with open(rag_dir, "w", encoding="utf-8") as f:
    json.dump(qa_data_list, f, ensure_ascii=False, indent=2)

metrics_summary = {
    "total_count": total_count,
    "correct_count": correct_count,
    "accuracy": accuracy,
    "average_score": average_score,
    "micro_precision": micro_prec,
    "micro_recall": micro_rec,
    "micro_f1": micro_f1,
    "macro_precision": macro_prec,
    "macro_recall": macro_rec,
    "macro_f1": macro_f1
}
metrics_path = os.path.join(output_dataset_dir, f"{dataset_name}_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

logging.info(f"📊 指标已保存到 {metrics_path}")
joint.close()
