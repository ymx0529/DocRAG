import os
import sys
import json
import logging
import time
import re
import pandas as pd
from sentence_transformers import SentenceTransformer 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from typing import Set, Any

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

############################################# 辅助函数 ###############################################
def parse_evidence_pages(evidence_field) -> Set[int]:
    """
    兼容 evidence_pages 的多种格式：
      - [15, 16]
      - "[19, 20]"
      - "19,20"
      - "19"
      - None -> set()
    返回 set[int]
    """
    if evidence_field is None:
        return set()
    if isinstance(evidence_field, list):
        out = set()
        for x in evidence_field:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    if isinstance(evidence_field, int):
        return {int(evidence_field)}
    if isinstance(evidence_field, str):
        s = evidence_field.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return set(int(x) for x in parsed if str(x).isdigit())
            if isinstance(parsed, int):
                return {parsed}
        except Exception:
            nums = re.findall(r'\d+', s)
            return set(int(x) for x in nums)
    try:
        return {int(evidence_field)}
    except Exception:
        return set()

def get_gold_pages(qa_item: dict) -> Set[int]:
    """
    从 qa_item 中尝试读取可能的标注页码字段，统一减 1（因为 gold 从 1 开始）。
    """
    candidates = [
        "evidence_pages", "evidence", "evidence_page", "evidence_page_ids",
        "pages", "ground_truth_pages", "evidencePages", "evidencePage"
    ]
    for k in candidates:
        if k in qa_item:
            raw = parse_evidence_pages(qa_item.get(k))
            return set(x - 1 for x in raw if isinstance(x, int))
    return set()

def _extract_pages_from_obj(obj: Any) -> Set[int]:
    """
    递归尝试从任意对象中提取页码。
    支持 int / list / tuple / dict / str（JSON或数字）/ 嵌套结构。
    """
    pages = set()
    if obj is None:
        return pages
    if isinstance(obj, int):
        pages.add(int(obj))
        return pages
    if isinstance(obj, (list, tuple)):
        for e in obj:
            pages |= _extract_pages_from_obj(e)
        return pages
    if isinstance(obj, dict):
        for key in ("page", "page_idx", "page_id", "pageno", "page_num", "pageNo"):
            if key in obj:
                try:
                    pages.add(int(obj[key]))
                    return pages
                except Exception:
                    pass
        for v in obj.values():
            pages |= _extract_pages_from_obj(v)
        return pages
    if isinstance(obj, str):
        s = obj.strip()
        try:
            parsed = json.loads(s)
            return _extract_pages_from_obj(parsed)
        except Exception:
            nums = re.findall(r'\d+', s)
            if nums:
                try:
                    pages.add(int(nums[0]))
                except Exception:
                    pass
            return pages
    try:
        if hasattr(obj, "__int__"):
            pages.add(int(obj))
    except Exception:
        pass
    return pages

def predicted_pages_from_chunks(chunks: dict) -> Set[int]:
    """
    从 chunks 中提取预测页码集合。
    支持多种格式：
      - {"text": {(page_idx, para_idx): {...}}, "multimodal": {...}}
      - {"text": {"(12,3)": {...}}, ...}
      - {"text": [{"page":12, ...}, ...], ...}
    """
    pages = set()
    if not chunks:
        return pages
    if isinstance(chunks, dict):
        for section_key, section_val in chunks.items():
            pages |= _extract_pages_from_obj(section_key)
            if isinstance(section_val, dict):
                for k, v in section_val.items():
                    pages |= _extract_pages_from_obj(k)
                    pages |= _extract_pages_from_obj(v)
            elif isinstance(section_val, (list, tuple)):
                for entry in section_val:
                    pages |= _extract_pages_from_obj(entry)
            else:
                pages |= _extract_pages_from_obj(section_val)
    elif isinstance(chunks, (list, tuple)):
        for entry in chunks:
            pages |= _extract_pages_from_obj(entry)
    else:
        pages |= _extract_pages_from_obj(chunks)
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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
log_dir = Path(__file__).parent / "Logs/Log2"
logger = get_logger(log_dir=log_dir, backup_count=10)
logging.info("\n------ 系统启动 ------\n")

########################################### 大模型API配置 ############################################
max_token_count = 32000
model="qwen-plus" 
tokenizer = "Qwen/Qwen3-4B-Instruct-2507"
reasoning_model = False
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
vl_model = "qwen-vl-plus"
api_key="sk-5e1abc2217e5450faaf791d260ed6074"
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

SPACE = "mrag_sub2"
joint = JointHandler(space_name=SPACE, 
                     nebula_host="127.0.0.1", nebula_port=9669, 
                     milvus_host="127.0.0.1", milvus_port="19530", 
                     collection_name="mrag_sub2")
joint.setup(embedding_dim=encoder.get_sentence_embedding_dimension())

############################################## 路径设置 ###############################################
dataset_name = "SlideVQA"
ocr_dir = os.path.join(f"/home/hdd/MRAG/Dataset/{dataset_name}")
output_dir = os.path.join(f"Generation/Output")
output_dataset_dir = os.path.join(f"Generation/Output/{dataset_name}")
os.makedirs(output_dataset_dir, exist_ok=True)
rag_dir = os.path.join("Generation/Output", f"{dataset_name}_rag.json")
os.makedirs(os.path.dirname(rag_dir), exist_ok=True)
judge_dir = os.path.join("Generation/Output", f"{dataset_name}_judge.csv")

json_path = '/home/hdd/MRAG/Dataset/SlideVQA_QA.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

doc_qa_dict = {}
for item in data:
    pdf_name = item['deck_name']
    question = item['question']
    answer = item['answer']
    if pdf_name not in doc_qa_dict:
        doc_qa_dict[pdf_name] = []
    doc_qa_dict[pdf_name].append({
        "question": question,
        "answer": answer,
        **item
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
        gold_pages = get_gold_pages(qa)

        query = question
        q_vector = encoder.encode(query, convert_to_tensor=False)
        search_results = joint.search_neighbors_by_vector(q_vector.tolist(), top_k=5, partition=None)
        results_matched = get_related_entities(entity_id=search_results, graph_data=final_graph)

        default_params = {"alpha": 0.4, "beta": 0.3, "lam": 0.3}
        parameters = None
        try:
            rag_parameters = get_system_parameter(
                model=model, reasoning_model=reasoning_model,
                query=query, api_key=api_key, base_url=base_url
            )
            if isinstance(rag_parameters, list) and rag_parameters:
                rag_parameters = rag_parameters[0]
            if isinstance(rag_parameters, dict):
                parameters = extract_weights(rag_parameters)
        except Exception:
            parameters = default_params
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

# ===== 汇总统计 =====
accuracy = correct_count / total_count if total_count > 0 else 0.0
average_score = score_sum / total_count if total_count > 0 else 0.0

total_tp = total_fp = total_fn = 0
for _, row in results_df.iterrows():
    pset = set(row["predicted_pages"])
    gset = set(row["gold_pages"])
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
