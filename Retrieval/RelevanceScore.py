import ast
import csv
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

def _graph_entities_loader(graph_data: list) -> list:
    """
    载入 JSON 格式的 graph 并筛选出实体部分
    返回:
        实体列表
    """
    entities = []
    for item in graph_data:
        # 判断是不是实体
        if all(key in item for key in ["name", "type", "description", "entityID"]):
            entities.append(item)
    
    return entities

def _compute_entity_similarities(entities: list, 
                                 query: str, 
                                 encoder: SentenceTransformer
                                 ) -> dict:
    """
    计算query和子图中各个entity的相似度。
    
    参数:
        entities: 实体列表，包含 name、type、description、entityID
        query: 查询
        encoder: 编码模型 用于获取语义向量
    
    返回:
        similarities: {entityID (str): 相似度分数}
    """
    # 拼接实体文本
    entity_texts = [
        f"{e['name']} {e['type']} {e['description']}" for e in entities if "entityID" in e
    ]
    entity_ids = [e["entityID"] for e in entities if "entityID" in e]

    # 编码并计算余弦相似度
    query_embedding = encoder.encode(query, convert_to_tensor=True)
    entity_embeddings = encoder.encode(entity_texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, entity_embeddings)[0]

    similarities = {
        entity_ids[i]: float(cos_scores[i]) for i in range(len(entity_ids))
    }

    return similarities

def _chunk_statistics(entities:list) -> tuple[dict, dict, dict, dict]:
    """
    从JSON格式的实体合集中提取实体的 entityID 前两位 (chunkID)
    统计唯一值及出现次数

    返回:
        text_entities: {chunkID: [entityID列表]} 文本chunk具体实体
        multimodal_entities: {chunkID: [entityID列表]} 多模态chunk具体实体
    """
    text_chunk = []
    multimodal_chunk = []
    text_entities = defaultdict(list)
    multimodal_entities = defaultdict(list)

    for item in entities:
        if "entityID" in item:
            entity_id = ast.literal_eval(item["entityID"])
            chunk_id_list = ast.literal_eval(item["chunkID"])

            if isinstance(entity_id[2], (int, float)):  # 文本chunk
                text_chunk.append(chunk_id_list)
                for cid in chunk_id_list:
                    text_entities[tuple(cid)].append(str(entity_id)) # 转成str
            elif (isinstance(entity_id[2], str) and entity_id[2]!="anchor"):  # 多模态chunk
                multimodal_chunk.append(chunk_id_list)
                for cid in chunk_id_list:
                    multimodal_entities[tuple(cid)].append(str(entity_id))
    # 统计并转成普通字典
    return dict(text_entities), dict(multimodal_entities)

def _load_pagerank_dict(csv_file_path: str) -> dict:
    """
    读取CSV文件 将 _id 和 _pagerank 组成字典。
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        dict: {_id(str): _pagerank(float)}
    """
    pagerank_dict = {}
    try:
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pagerank_dict[row["_id"]] = float(row["_pagerank"])
    except FileNotFoundError:
        print(f"未找到CSV文件: {csv_file_path}")
    except Exception as e:
        print(f"读取CSV文件出错: {e}")

    return pagerank_dict

def _load_closeness_dict(csv_file_path: str) -> dict:
    """
    读取CSV文件 将 _id 和 _closeness 组成字典。
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        dict: {_id(str): _closeness(float)}
    """
    closeness_dict = {}
    try:
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                closeness_dict[row["_id"]] = float(row["_closeness"])
    except FileNotFoundError:
        print(f"未找到CSV文件: {csv_file_path}")
    except Exception as e:
        print(f"读取CSV文件出错: {e}")

    return closeness_dict

def _relevance_score(similarities: dict, 
                     chunk_entities: dict, 
                     alpha: float
                     ) -> dict:
    """
    计算每个 chunk 对应的节点的相似度数值和，和节点数量，并转换为概率值，
    根据权重，计算每个 chunk 的综合相关性得分 Score(c)。

    参数:
        similarities: {entityID(str): similarity_score(float)}
        chunk_entities: {chunkID(tuple): [entityID(str), ...]} 每个chunk包含的entity
        alpha: 相似度和节点数量奖励的权重, 取值范围 [0, 1]
    返回:
        {chunkID(tuple): score(float)}
    """
    sim_scores = defaultdict(float)
    num_socres = defaultdict(float)
    for chunk_id, entity_list in chunk_entities.items():
        for eid in entity_list:
            sim = similarities.get(eid, 0.0)    # 没有则视为0
            sim_scores[chunk_id] += sim
            num_socres[chunk_id] += 1

    # 归一化为和为1的概率值
    total1 = sum(sim_scores.values())
    similarities_prob = {node: val/total1 for node, val in sim_scores.items()}
    total2 = sum(num_socres.values())
    chunk_entities_prob = {node: val/total2 for node, val in num_socres.items()}

    chunk_scores = {node: alpha * similarities_prob.get(node, 0.0) + (1-alpha) * chunk_entities_prob.get(node, 0.0)
                    for node in sim_scores.keys()}

    return chunk_scores

def _structure_score(pagerank_dict: dict, 
                     closeness_dict: dict, 
                     chunk_entities: dict, 
                     beta: float
                     ) -> dict:
    """
    根据每个 chunk 对应节点的 PageRank 值和 Closeness Centrality 值，
    根据权重，计算每个 chunk 的结构得分 Score(c)。
    参数:
        pagerank_dict: {entityID(str): pagerank(float)}
        closeness_dict: {entityID(str): closeness(float)}
        chunk_entities: {chunkID(tuple): [entityID(str),...]} 每个chunk包含的entity
        beta: PageRank 和 Closeness Centrality 奖励的权重, 取值范围 [0, 1]
    """
    
    # closeness 值归一化为和为1的概率值，和 pagerank 归一到同一量级
    total = sum(closeness_dict.values())
    closeness_prob_dict = {node: val/total for node, val in closeness_dict.items()}

    chunk_scores = {}
    for chunk_id, entity_list in chunk_entities.items():
        total = 0.0
        for eid in entity_list:
            pagerank = pagerank_dict.get(eid, 0.0)   # 没有则视为0
            closeness = closeness_prob_dict.get(eid, 0.0)    # 没有则视为0
            total += beta * pagerank + (1-beta) * closeness
        chunk_scores[chunk_id] = total

    return chunk_scores

def _rank_scores(dict1, dict2, lambda_value):
    """
    对两个字典进行排名，返回一个新的字典，其中键是原始字典的键，值是排名。
    
    参数：
    - dict1: 第一个字典
    - dict2: 第二个字典
    
    返回：
    - final_rank_dict: key 对应的综合排名字典
    """
    # 对 dict 排名，值越大排名值越大
    sorted_items1 = sorted(dict1.items(), key=lambda x: x[1])
    rank1 = {k: i+1 for i, (k, _) in enumerate(sorted_items1)}
    
    sorted_items2 = sorted(dict2.items(), key=lambda x: x[1])
    rank2 = {k: i+1 for i, (k, _) in enumerate(sorted_items2)}
    
    # 对每个 key 的排名相加
    final_rank_dict = {k: lambda_value * rank1[k] + 
                       (1-lambda_value) * rank2[k] for k in dict1.keys()}
    
    return final_rank_dict

def chunk_score(query: str, 
                graph_data: list, 
                pagerank: dict | str,   # 允许传 dict / str(csv路径) / None
                closeness: dict | str, 
                encoder_model: SentenceTransformer, 
                alpha=0.5, 
                beta=0.5,  
                lam=0.5
                )-> tuple[dict, dict]:
    """
    对text和多模态段落, 分别计算每个chunk的相关性得分 RelevanceScore(c)
    参数:
        query: 查询
        graph_data: 图数据
        pagerank: PageRank 数据，可以是 dict 或 csv 文件路径
        closeness: Closeness Centrality 数据，可以是 dict 或 csv 文件路径
        encoder_model: 编码模型 用于获取语义向量
        alpha: 相关性权重
        beta: 全图重要性权重
        lam: 综合权重
    返回:
        {chunkID(tuple): score(float)}, 
        {chunkID(tuple): score(float)}
    """
    # 1. 提取实体
    entities = _graph_entities_loader(graph_data)
    # 2. 统计chunk对应的实体
    text_chunk_entities, multimodal_chunk_entities = _chunk_statistics(entities)
    # 3. 计算被选中的实体与 query 的相似度，余弦相似度范围大体为0~1
    similarities = _compute_entity_similarities(entities, query, encoder_model)

    # 4. 处理 PageRank 输入 和 Closeness Centrality 输入
    if isinstance(pagerank, dict):
        pagerank_dict = pagerank
    elif isinstance(pagerank, str):  # 传的是路径
        pagerank_dict = _load_pagerank_dict(pagerank)
    else:
        pagerank_dict = {}
    if isinstance(closeness, dict):
        closeness_dict = closeness
    elif isinstance(closeness, str):
        closeness_dict = _load_closeness_dict(closeness)
    else:
        closeness_dict = {}

    # 5. 计算 chunk 的 relevance score
    relevance_scores_text = _relevance_score(similarities, text_chunk_entities, alpha)
    relevance_scores_multimodal = _relevance_score(similarities, multimodal_chunk_entities, alpha)

    # 6. 计算 chunk 的 structure score
    structure_scores_text = _structure_score(pagerank_dict, closeness_dict, text_chunk_entities, beta)
    structure_scores_multimodal = _structure_score(pagerank_dict, closeness_dict, multimodal_chunk_entities, beta)

    # 7. 综合排序
    scores_text = _rank_scores(relevance_scores_text, structure_scores_text, lam)
    scores_multimodal = _rank_scores(relevance_scores_multimodal, structure_scores_multimodal, lam)

    return scores_text, scores_multimodal

