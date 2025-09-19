import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def _select_top_k_chunks(scores: dict, top_k: int = 5) -> list[str]:
    """
    从 scores 字典中，按 value 排序，选取前 top_k 个 chunkID。

    参数:
        scores: {chunkID: score}
        top_k: 选取数量
    返回:
        [chunkID1, chunkID2, ...] 排好序的前k个结果
    """
    # 按value降序排序
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_id_list = [chunk_id for chunk_id, _ in sorted_items[:top_k]]
    return top_id_list

def _load_chunks(file_path: Path, 
                 pdf_name: str,
                 text_chunkID: list[str], 
                 multimodal_chunkID: list[str]
                 )-> dict:
    """
    读取指定文件夹中最新的 OCR 解析文件，并根据统计字典提取对应的三元组
    text_chunkID 里也可能包含 caption 等多模态来源的文本
    这里会详细区分原始chunk是否是多模态节点
    
    参数:
    file_path: JSON 文件的文件夹路径
    text_chunkID: 文本 chunkID list [(page_idx, number), ...]
    multimodal_chunkID: 多模态 chunkID list [(page_idx, number), ...]

    返回:
        chunks = {
            "text": {(page_idx, 序号): 对应json字典},
            "multimodal": {(page_idx, 序号): 对应json字典}
        }
    """
    json_files = list(file_path.glob("*.json"))
    if not json_files:
        print("未找到JSON文件")
        return defaultdict(list), ""
    # 取最 JSON 文件
    json_file_path = os.path.join(file_path, f"{pdf_name}_content_list.json")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按页码分组
    pages = defaultdict(list)
    for item in data:
        pages[item['page_idx']].append(item)

    chunks = {"text": {}, "multimodal": {}}
    # 遍历 text_chunkID
    for (page_idx, paragraph_idx) in text_chunkID:
        if page_idx in pages and paragraph_idx < len(pages[page_idx]):
            chunk = pages[page_idx][paragraph_idx]
            if chunk.get("type") == "text":  # text来源的文本
                chunks["text"][(page_idx, paragraph_idx)] = pages[page_idx][paragraph_idx]
            else:                            # caption等来源的文本
                chunks["multimodal"][(page_idx, paragraph_idx)] = pages[page_idx][paragraph_idx]
    # 遍历 multimultimodal_chunkIDmodal_stats
    for (page_idx, paragraph_idx) in multimodal_chunkID:
        if page_idx in pages and paragraph_idx < len(pages[page_idx]):
            chunks["multimodal"][(page_idx, paragraph_idx)] = pages[page_idx][paragraph_idx]
    
    return chunks

def chunk_loader(ocr_json_path: Path, 
                 pdf_name: str,
                 scores_text: dict, 
                 scores_multimodal: dict, 
                 top_k_text: int = 5, 
                 top_k_multimodal: int = 4
                 ) -> dict:
    """"
    ocr_json_path: OCR 解析文件的文件夹路径
    scores_text: 文本来源的 chunk 得分字典 {chunkID: score}
    scores_multimodal: 多模态来源 chunk 得分字典 {chunkID: score}
    top_k_text: 选取文本来源的 top_k 个chunk
    scores_multimodal: 选取多模态来源的 top_k 个 chunk
    返回:
        chunks = {
            "text": {(page_idx, 序号): 对应json字典},
            "multimodal": {(page_idx, 序号): 对应json字典}
        }
    """
    # 按value降序排序
    top_text_chunkID = _select_top_k_chunks(scores_text, top_k_text)
    top_multimodal_chunkID = _select_top_k_chunks(scores_multimodal, top_k_multimodal)

    # 根据段落type分别加载对应的段落内容
    chunks = _load_chunks(ocr_json_path, pdf_name, top_text_chunkID, top_multimodal_chunkID)

    return chunks

def get_related_entities(entity_id: dict, 
                         graph_data: List[Dict], 
                         hop: int = 1
                         ) -> List[Dict]:
    """
    根据检索到的 entityID ，从知识图谱中获取与之相关的实体
    1. 只保留 entity ，并剔除掉 anchor节点
    2. 只考虑 1-hop 和 2-hop 关系
    参数：
        entity_id (dict): 目标实体ID。
        graph_data (list): 包含实体和关系的知识图谱数据。
        hop (int): 跳数 默认为1。
    返回：
        list: 实体的列表。
    """
    # 构建实体ID到实体的映射
    combined = {"1-hop": [], "2-hop": []}
    for value in entity_id.values():
        combined["1-hop"].extend(value["1-hop"])
        combined["2-hop"].extend(value["2-hop"])
    combined["2-hop"].extend(combined["1-hop"])

    # 只保留 entity ，并剔除掉 anchor节点
    entity_dict = {
        e["entityID"]: e
        for e in graph_data
        if "entityID" in e and e.get("type", "") != "SEGMENT ANCHOR NODE"
        }

    # 根据跳数获取相关实体
    hop_key = f"{hop}-hop"
    results = [entity_dict[eid] for eid in combined.get(hop_key, []) if eid in entity_dict]

    return results
