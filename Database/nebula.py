from typing import Dict, List
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import ttypes
import time


# 预定义 TAG 名称
PREDEFINED_TAGS = [
    "ENTITY_NODE",
    "SEGMENT ANCHOR NODE",
    "ORIGINAL_IMAGE",
    "EQUATION_NODE",
    "TABLE_NODE",
]

ENTITY_TAG = PREDEFINED_TAGS[0]
SPECIAL_TAGS = set(PREDEFINED_TAGS[1:])

class NebulaHandler:
    """
    Nebula handler:
    - 固定五个预定义 TAG
    - ENTITY_NODE 顶点 vid_type=FIXED_STRING(64)，使用 entityID 作为 vid
    - ENTITY_NODE 附加字段：name, description, type, chunkID
    - 统一边类型：RelatesTo(relationship string, relationship_strength double)
    """

    def __init__(self, space_name: str = "mrag_test", host: str = "127.0.0.1", port: int = 9669, user: str = "root", password: str = "nebula"):
        self.space_name = space_name
        config = Config()
        self.pool = ConnectionPool()
        ok = self.pool.init([(host, port)], config)
        if not ok:
            raise RuntimeError("Nebula ConnectionPool init failed")
        self.session = self.pool.get_session(user, password)
        self._ensure_space_exists()

        # name -> entityID 映射
        self.name_to_vid: Dict[str, str] = {}

    def _normalize_vid(self, vid: str) -> str:
        """确保 Nebula vid 符合 FIXED_STRING(64) 要求"""
        if not isinstance(vid, str):
            vid = str(vid)

        # 🔹 如果超过 128，直接截断前 128 位
        if len(vid) > 128:
            return vid[:128]
        return vid

    """ def _exec(self, nGQL: str):
        res = self.session.execute(nGQL)
        if res.error_code() != ttypes.ErrorCode.SUCCEEDED:
            raise RuntimeError(f"Nebula nGQL failed: {nGQL}\n{res.error_msg()}")
        return res """
    
    def _exec(self, nGQL: str, retries: int = 20, retry_interval: float = 1.0):
        for attempt in range(retries):
            try:
                # 尝试执行 nGQL 查询
                res = self.session.execute(nGQL)
                if res.error_code() == ttypes.ErrorCode.SUCCEEDED:
                    return res
                if "Not the leader" in res.error_msg():
                    print(f"[WARN] Not the leader, 第 {attempt+1} 次重试 ...")
                else:
                    # 如果出现其他错误，打印并重试
                    print(f"[WARN] 执行失败，错误信息: {res.error_msg()}，第 {attempt+1} 次重试 ...")
                
            except Exception as e:
                # 捕获所有传输异常或其他所有异常，打印错误信息并重试
                print(f"[WARN] 异常: {str(e)}，第 {attempt+1} 次重试 ...")
            
            time.sleep(retry_interval)
        raise RuntimeError(f"[ERROR] 重试 {retries} 次后仍然失败: {nGQL}")

    def _ensure_space_exists(self):
        self._exec(
            f'CREATE SPACE IF NOT EXISTS {self.space_name}(partition_num=1, replica_factor=1, vid_type=FIXED_STRING(128));'
        )
        # 等待 space 出现在 SHOW SPACES
        for i in range(30):  # 最多等 30 秒
            res = self._exec("SHOW SPACES;")
            spaces = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            if self.space_name in spaces:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直未出现在 SHOW SPACES")
        self._exec(f'USE {self.space_name};')

    def create_schema(self):
        """
        确保 space/schema 存在，但不会删除已有数据
        """
        # 如果空间不存在则创建
        self._exec(
            f'CREATE SPACE IF NOT EXISTS {self.space_name}('
            f'partition_num=1, replica_factor=1, vid_type=FIXED_STRING(128));'
        )

        # 尝试 USE 空间
        for i in range(30):
            try:
                self._exec(f'USE {self.space_name};')
                print(f"[Nebula] Space {self.space_name} 已确认可用 ✅")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直无法 USE")

        # 创建 TAG & EDGE（不会覆盖已有数据）
        self._exec(
            f'CREATE TAG IF NOT EXISTS `{ENTITY_TAG}`('
            f'name string, description string, type string, chunkID string);'
        )
        for tag in SPECIAL_TAGS:
            self._exec(
                f'CREATE TAG IF NOT EXISTS `{tag}`(name string, description string, chunkID string);'
            )
        self._exec(
            'CREATE EDGE IF NOT EXISTS RelatesTo(relationship string, relationship_strength double);'
        )

        # 确认 TAG 已存在
        for i in range(30):
            res = self._exec("SHOW TAGS;")
            tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            if "ENTITY_NODE" in tags:
                print("[Nebula] ENTITY_NODE schema 已确认存在 ✅")
                break
            time.sleep(1)
        else:
            raise RuntimeError("ENTITY_NODE schema 未成功创建")



    def reset_space(self):
        print(f"[DEBUG] 尝试删除空间 {self.space_name}")
        self._exec(f'DROP SPACE IF EXISTS {self.space_name};')
        time.sleep(2)

        print(f"[DEBUG] 重新创建空间 {self.space_name}")
        self._exec(
            f'CREATE SPACE IF NOT EXISTS {self.space_name}('
            f'partition_num=1, replica_factor=1, vid_type=FIXED_STRING(128));'
        )

        # 等待 space 出现在 SHOW SPACES
        for i in range(30):
            res = self._exec("SHOW SPACES;")
            spaces = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            if self.space_name in spaces:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直未出现在 SHOW SPACES")

        # 再等待 USE 成功
        for i in range(30):
            try:
                self._exec(f'USE {self.space_name};')
                print(f"[Nebula] Space {self.space_name} 已确认可用 ✅")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直无法 USE")

        # 🔹 创建 TAG & EDGE
        self._exec(
            f'CREATE TAG IF NOT EXISTS `{ENTITY_TAG}`('
            f'name string, description string, type string, chunkID string);'
        )
        for tag in SPECIAL_TAGS:
            self._exec(
                f'CREATE TAG IF NOT EXISTS `{tag}`(name string, description string, chunkID string);'
            )
        self._exec(
            'CREATE EDGE IF NOT EXISTS RelatesTo(relationship string, relationship_strength double);'
        )

        # 🔹 确认 TAG 生效
        for i in range(30):
            res = self._exec("SHOW TAGS;")
            tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            print("[DEBUG] 当前 TAGS:", tags)
            if "ENTITY_NODE" in tags:
                print("[Nebula] ENTITY_NODE schema 已确认创建成功 ✅")
                break
            time.sleep(1)
        else:
            raise RuntimeError("ENTITY_NODE schema 未成功创建")

        # 🔹 强制触发所有 TAG 的 schema 同步
        for tag in [ENTITY_TAG] + list(SPECIAL_TAGS):
            for i in range(10):
                try:
                    self._exec(f'DESC TAG `{tag}`;')
                    print(f"[Nebula] DESC TAG {tag} 成功，Storage 已同步 ✅")
                    break
                except Exception as e:
                    print(f"[DEBUG] DESC TAG {tag} 第 {i+1} 次失败，等待同步: {e}")
                    time.sleep(1)
            else:
                raise RuntimeError(f"{tag} schema 一直未同步到 Storage")

        # 🔹 新增: 强制触发 EDGE 的 schema 同步
        for i in range(10):
            try:
                self._exec('DESC EDGE RelatesTo;')
                print("[Nebula] DESC EDGE RelatesTo 成功，Storage 已同步 ✅")
                break
            except Exception as e:
                print(f"[DEBUG] DESC EDGE RelatesTo 第 {i+1} 次失败，等待同步: {e}")
                time.sleep(1)
        else:
            raise RuntimeError("RelatesTo edge schema 一直未同步到 Storage")
        
        #time.sleep(10) 

        # Dummy 节点写入测试（静默重试，直到成功）
        test_vid = self._normalize_vid("DummyTestNode001")   # 固定测试点 ID
        last_err = None

        for i in range(60):  # 最多重试 60 次，等待1分钟
            try:
                nGQL = (
                    f'INSERT VERTEX `{ENTITY_TAG}`(name, description, type, chunkID) '
                    f'VALUES "{test_vid}":("tmp_name", "tmp_desc", "tmp_type", "0");'
                )
                self._exec(nGQL)
                self._exec(f'FETCH PROP ON `{ENTITY_TAG}` "{test_vid}" YIELD properties(vertex);')
                self._exec(f'DELETE VERTEX "{test_vid}";')  # 删除测试点
                print(f"[Nebula] Dummy 节点写入成功 ✅ (第 {i+1} 次尝试)")
                break
            except Exception as e:
                msg = str(e)
                last_err = msg
                time.sleep(1)
                continue
        else:
            print(f"[WARN] Dummy 节点写入在 20 次尝试后仍未成功，最后错误: {last_err}")



    # ---------- 实体写入 ----------
    def insert_entity(self, entityID: str, type_name: str, name: str, description: str = "", chunkID: str = ""):
        self._exec(f'USE {self.space_name};')
        
        # 🔹 修改点 : 每次插入前强制 USE
        self._exec(f'USE {self.space_name};')

        # Step1: 确认 TAG 是否存在
        res = self._exec("SHOW TAGS;")
        tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
        if "ENTITY_NODE" not in tags:
            raise RuntimeError("[ERROR] ENTITY_NODE schema 未找到，请确认 reset_space 是否成功执行")

        target_tag = type_name if type_name in SPECIAL_TAGS else ENTITY_TAG

        # 清洗
        safe_desc = description.replace('\\n', ' ').replace('\\r', ' ').replace('\\', '\\\\').replace('"', '\\"')
        safe_name = name.replace('\\n', ' ').replace('\\r', ' ').replace('\\', '\\\\').replace('"', '\\"')
        safe_chunk = chunkID.replace('\\n', ' ').replace('\\r', ' ').replace('\\', '\\\\').replace('"', '\\"')

        if target_tag == ENTITY_TAG:
            nGQL = (
                f'INSERT VERTEX `{ENTITY_TAG}`(name, description, type, chunkID) '
                f'VALUES "{entityID}":("{safe_name}", "{safe_desc}", "{type_name}", "{safe_chunk}");'
            )

        elif target_tag == PREDEFINED_TAGS[1]:
            nGQL = (
                f'INSERT VERTEX `{target_tag}`(name, description) '
                f'VALUES "{entityID}":("{safe_name}", "{safe_desc}");'
            )

        else:
            nGQL = (
                f'INSERT VERTEX `{target_tag}`(name, description, chunkID) '
                f'VALUES "{entityID}":("{safe_name}", "{safe_desc}", "{safe_chunk}");'
            )

        self._exec(nGQL)
        time.sleep(0.05)
        verify = self._exec(f'FETCH PROP ON `{target_tag}` "{entityID}" YIELD properties(vertex);')
        print(f"[Nebula] Inserted `{target_tag}` {entityID}: ", [r.values for r in verify.rows()])

        # 建立映射：name -> entityID
        self.name_to_vid[name] = entityID

    # ---------- 边写入 ----------
    def insert_relation(self, src: str, dst: str,
                        relationship: str, relationship_strength: float):
        self._exec(f'USE {self.space_name};')

        def sanitize(s: str) -> str:
            return s.replace('\\n', ' ').replace('\\r', ' ').replace('\\', '\\\\').replace('"', '\\"')

        src, dst, relationship = sanitize(src), sanitize(dst), sanitize(relationship)

        # 如果传入的是 name，用映射转成 vid
        src_vid = self.name_to_vid.get(src, src)
        dst_vid = self.name_to_vid.get(dst, dst)

        print(f"[Nebula] Inserting edge: raw {src} -> {dst} => resolved {src_vid} -> {dst_vid}")

        nGQL = (
            f'INSERT EDGE RelatesTo(relationship, relationship_strength) '
            f'VALUES "{src_vid}"->"{dst_vid}":("{relationship}", {float(relationship_strength)});'
        )
        self._exec(nGQL)
        time.sleep(0.05)
        verify = self._exec(
            f'FETCH PROP ON RelatesTo "{src_vid}"->"{dst_vid}" YIELD properties(edge);'
        )
        print(f"[Nebula] Inserted Relation {src_vid}->{dst_vid}: ", [r.values for r in verify.rows()])

    
    # -------------------  批量写入（新增部分） -------------------
    def insert_entities_bulk(self, entities: List[Dict], batch_size: int = 1000):
        """批量插入实体"""
        self._exec(f'USE {self.space_name};')

        #  确认 TAG 是否存在
        res = self._exec("SHOW TAGS;")
        tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
        print("[DEBUG] insert_entities_bulk 当前 TAGS:", tags)
        if "ENTITY_NODE" not in tags:
            raise RuntimeError("[ERROR] ENTITY_NODE schema 未找到，无法批量插入")

         #  二次校验 TAG 是否存在
        tags = self._exec("SHOW TAGS;")
        tags_list = [row.values[0].get_sVal().decode("utf-8") for row in tags.rows()]
        if ENTITY_TAG not in tags_list:
            raise RuntimeError("ENTITY_NODE TAG 不存在，请确认 reset_space 是否成功执行")
        
        buffer = {}  # 不同 TAG 分开缓存

        for e in entities:
            raw_id = e["entityID"]
            entityID = self._normalize_vid(raw_id)   # ✅ 使用规范化 vid

            original_name = e.get("name", "")
            self.name_to_vid[original_name] = entityID

            type_name = e.get("type", "")
            name = e.get("name", "").replace('"', '\\"')
            desc = e.get("description", "").replace('"', '\\"')
            chunkID = e.get("chunkID", "").replace('"', '\\"')

            target_tag = type_name if type_name in SPECIAL_TAGS else ENTITY_TAG

            if target_tag == ENTITY_TAG:
                values = f'"{entityID}":("{name}", "{desc}", "{type_name}", "{chunkID}")'
            elif target_tag in SPECIAL_TAGS:
                # special TAG 没有 type 字段
                values = f'"{entityID}":("{name}", "{desc}", "{chunkID}")'
            else:
                continue

            if target_tag not in buffer:
                buffer[target_tag] = []
            buffer[target_tag].append(values)

            # 分批插入
            if len(buffer[target_tag]) >= batch_size:
                self._flush_entities(buffer[target_tag], target_tag)
                buffer[target_tag].clear()

        # 插入剩余的
        for tag, values in buffer.items():
            if values:
                self._flush_entities(values, tag)

    
    def _flush_entities(self, values: List[str], tag: str, retries: int = 5, retry_interval: float = 1.0):
        """批量插入实体"""
        if tag == ENTITY_TAG:
            nGQL = f'INSERT VERTEX `{ENTITY_TAG}`(name, description, type, chunkID) VALUES {",".join(values)};'
        else:
            nGQL = f'INSERT VERTEX `{tag}`(name, description, chunkID) VALUES {",".join(values)};'

        last_err = None
        for _ in range(retries):
            try:
                self._exec(nGQL)
                print(f"[Nebula] Bulk inserted {len(values)} entities into {tag}")
                return  #  成功立即返回
            except Exception as e:
                msg = str(e)
                last_err = msg
                if "Not the leader" in msg or "Storage Error" in msg:
                    time.sleep(retry_interval)
                    continue
                raise  # 其他错误直接抛出
        raise RuntimeError(f"[ERROR] 插入 {tag} 批量数据失败超过 {retries} 次，最后错误: {last_err}")

    def insert_relations_bulk(self, relations: List[Dict], batch_size: int = 1000):
        """批量插入边"""
        
        values = []
        for r in relations:
            raw_src = r["source"]
            raw_dst = r["target"]

            src_vid = self.name_to_vid.get(raw_src, raw_src)
            dst_vid = self.name_to_vid.get(raw_dst, raw_dst)
            src = self._normalize_vid(src_vid)
            dst = self._normalize_vid(dst_vid)

            rel = r.get("relationship", "").replace('"', '\\"')
            strength = float(r.get("relationship_strength", 0.0))

            values.append(f'"{src}"->"{dst}":("{rel}", {strength})')
           
            if len(values) >= batch_size:
                self._flush_relations(values)
                values.clear()

        if values:
            self._flush_relations(values)
        

    def _flush_relations(self, values: List[str], retries: int = 5, retry_interval: float = 1.0):
        """批量插入关系"""
        self._exec(f'USE {self.space_name};')
        nGQL = f'INSERT EDGE RelatesTo(relationship, relationship_strength) VALUES {",".join(values)};'
        

        last_err = None
        for _ in range(retries):
            try:
                self._exec(nGQL)
                print(f"[Nebula] Bulk inserted {len(values)} relations")  #  成功打印
                return
            except Exception as e:
                msg = str(e)
                last_err = msg
                time.sleep(retry_interval)
                continue
        
        raise RuntimeError(f"[ERROR] 插入关系批量数据失败超过 {retries} 次，最后错误: {last_err}")


        # ---------- 检索邻居 ----------
    def fetch_neighbors_2_hops(self, entityID: str) -> Dict[str, List[str]]:
        self._exec(f'USE {self.space_name};')

        q1 = f'GO 1 STEPS FROM "{entityID}" OVER RelatesTo YIELD DISTINCT dst(edge) as dst;'
        r1 = self._exec(q1)
        hop1 = {row.values[0].get_sVal().decode("utf-8") for row in r1.rows()}
        hop1.add(entityID)  # 加入自身
        hop1 = sorted(hop1)

        q2 = f'GO 2 STEPS FROM "{entityID}" OVER RelatesTo YIELD DISTINCT dst(edge) as dst;'
        r2 = self._exec(q2)
        hop2_raw = {row.values[0].get_sVal().decode("utf-8") for row in r2.rows()}
        hop2 = sorted([v for v in hop2_raw if v != entityID and v not in hop1])

        print(f"[Nebula] Neighbors for {entityID}: 1-hop={hop1}, 2-hop={hop2}")
        return {"1-hop": hop1, "2-hop": hop2}

    def fetch_neighbors_1_hop(self, entityID: str) -> Dict[str, List[str]]:
        self._exec(f'USE {self.space_name};')

        q1 = f'GO 1 STEPS FROM "{entityID}" OVER RelatesTo YIELD DISTINCT dst(edge) as dst;'
        r1 = self._exec(q1)
        hop1 = {row.values[0].get_sVal().decode("utf-8") for row in r1.rows()}
        hop1.add(entityID)  # 加入自身
        hop1 = sorted(hop1)

        print(f"[Nebula] Neighbors for {entityID}: 1-hop={hop1}")
        return {"1-hop": hop1}


    def close(self):
        try:
            self.session.release()
        finally:
            self.pool.close()
    
    def compute_and_export_algorithms(
        self,
        pagerank_csv: str = "pagerank.csv",
        closeness_csv: str = "closeness.csv",
        pagerank_alpha: float = 0.85,
        pagerank_tol: float = 1e-6,
        pagerank_max_iter: int = 100,
        use_edge_weight: bool = False,
        closeness_undirected: bool = True,
        warn_node_threshold: int = 200000,
    ):
        """
        使用 MATCH 抓取所有节点和边，计算 PageRank 和 Closeness，
        输出两个 CSV 文件（vid,value）。
        """
        import networkx as nx
        import pandas as pd

        self._exec(f'USE {self.space_name};')

        # 1) 获取所有节点 id
        nodes = set()
        tags_res = self._exec("SHOW TAGS;")
        tags = []
        for row in tags_res.rows():
            try:
                tags.append(row.values[0].get_sVal().decode("utf-8"))
            except Exception:
                tags.append(str(row.values[0]))

        for tag in tags:
            q = f'MATCH (v:`{tag}`) RETURN id(v);'
            r = self._exec(q)
            for row in r.rows():
                try:
                    vid = row.values[0].get_sVal().decode("utf-8")
                except Exception:
                    vid = str(row.values[0])
                nodes.add(vid)

        # 2) 获取所有 RelatesTo 边
        edges = []
        if use_edge_weight:
            q = (
                "MATCH (a)-[e:RelatesTo]->(b) "
                "RETURN id(a) AS src, id(b) AS dst, e.relationship_strength AS weight;"
            )
        else:
            q = "MATCH (a)-[e:RelatesTo]->(b) RETURN id(a) AS src, id(b) AS dst;"
        r = self._exec(q)
        for row in r.rows():
            src = row.values[0].get_sVal().decode("utf-8")
            dst = row.values[1].get_sVal().decode("utf-8")
            w = 1.0
            if use_edge_weight and len(row.values) > 2:
                try:
                    wval = row.values[2]
                    if hasattr(wval, "get_dVal"):
                        w = float(wval.get_dVal())
                    elif hasattr(wval, "get_fVal"):
                        w = float(wval.get_fVal())
                    elif hasattr(wval, "get_iVal"):
                        w = float(wval.get_iVal())
                    else:
                        sval = wval.get_sVal()
                        if sval is not None:
                            w = float(sval.decode("utf-8"))
                except Exception:
                    w = 1.0
            edges.append((src, dst, float(w)))

        # 3) 构建图
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for s, d, w in edges:
            if s not in G:
                G.add_node(s)
            if d not in G:
                G.add_node(d)
            G.add_edge(s, d, weight=w)

        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        print(f"[Nebula] Graph built: nodes={n_nodes}, edges={n_edges}")

        if n_nodes == 0:
            raise RuntimeError("Graph is empty (no nodes). Aborting algorithm computation.")
        if n_nodes > warn_node_threshold:
            print(f"[Warning] 节点数 {n_nodes} 超过阈值 {warn_node_threshold}，networkx 计算可能很慢或内存不足。")

        # 4) PageRank
        try:
            pr = nx.pagerank(
                G,
                alpha=pagerank_alpha,
                tol=pagerank_tol,
                max_iter=pagerank_max_iter,
                weight="weight" if use_edge_weight else None,
            )
        except Exception as e:
            print(f"[Nebula] PageRank（带权）失败，退回无权计算：{e}")
            pr = nx.pagerank(G, alpha=pagerank_alpha, tol=pagerank_tol, max_iter=pagerank_max_iter)

        # 5) Closeness
        if closeness_undirected:
            H = G.to_undirected()
            clos = nx.closeness_centrality(H)
        else:
            clos = nx.closeness_centrality(G)

        # 6) 输出 CSV
        pr_df = pd.DataFrame(list(pr.items()), columns=["_id", "_pagerank"]).sort_values("_pagerank", ascending=False)
        pr_df.to_csv(pagerank_csv, index=False, float_format="%.12g")

        clos_df = pd.DataFrame(list(clos.items()), columns=["_id", "_closeness"]).sort_values("_closeness", ascending=False)
        clos_df.to_csv(closeness_csv, index=False, float_format="%.12g")

        print(f"[Nebula] Saved PageRank -> {pagerank_csv} (rows={len(pr_df)})")
        print(f"[Nebula] Saved Closeness -> {closeness_csv} (rows={len(clos_df)})")

        return pr, clos


        # 查询节点数量
    def get_node_count(self):
        # 统计每个标签的节点数量
        query_entity_node = "MATCH (v:ENTITY_NODE) RETURN COUNT(v) AS entity_node_count"
        res_entity_node = self._exec(query_entity_node)
        entity_node_count = res_entity_node.rows()[0].values[0].get_iVal()

        query_segment_anchor_node = "MATCH (v:`SEGMENT ANCHOR NODE`) RETURN COUNT(v) AS segment_anchor_node_count"
        res_segment_anchor_node = self._exec(query_segment_anchor_node)
        segment_anchor_node_count = res_segment_anchor_node.rows()[0].values[0].get_iVal()

        query_original_image = "MATCH (v:`ORIGINAL_IMAGE`) RETURN COUNT(v) AS original_image_node_count"
        res_original_image = self._exec(query_original_image)
        original_image_node_count = res_original_image.rows()[0].values[0].get_iVal()

        return {
            "ENTITY_NODE": entity_node_count,
            "SEGMENT ANCHOR NODE": segment_anchor_node_count,
            "ORIGINAL_IMAGE": original_image_node_count
        }

    # 查询边数量
    def get_edge_count(self):
        query = "MATCH ()-[e:RelatesTo]->() RETURN COUNT(e)"
        res = self._exec(query)
        return res.rows()[0].values[0].get_iVal()
