# milvus.py
import re
from typing import List, Optional
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

DEFAULT_COLLECTION = "mrag_test"

class MilvusHandler:
    """
    Milvus handler:
    - 单集合：默认名可配置（默认'mrag_test'）
    - 字段：entityID(VARCHAR pk), vector(FLOAT_VECTOR, dim)
    - 支持按 partition (例如 type 名) 创建分区；如果插入时没有 partition，则写入 default partition
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION, host: str = "127.0.0.1", port: str = "19530"):
        connections.connect(alias="default", host=host, port=port)
        self.collection: Optional[Collection] = None
        self._dim: Optional[int] = None
        self.collection_name = collection_name

    def create_collection(self, name: Optional[str] = None, dim: int = 1024):
        if name:
            self.collection_name = name
        self._dim = dim
        if utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name)
            print(f"[Milvus] Using existing collection: {self.collection_name}")
            return

        fields = [
            FieldSchema(name="entityID", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Entity vectors; primary key entityID")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"[Milvus] Created collection: {self.collection_name} (dim={dim})")

    def reset_collection(self, embedding_dim: int = 1024):
        """显式清空并重新建 collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"[Milvus] Dropped existing collection: {self.collection_name}")

        fields = [
            FieldSchema(name="entityID", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        schema = CollectionSchema(fields, description="Entity vectors; primary key entityID")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"[Milvus] Reset collection: {self.collection_name} (dim=embedding_dim)")

    def _ensure_partition(self, partition_name: str):
        if partition_name and not self.collection.has_partition(partition_name):
            self.collection.create_partition(partition_name)
            print(f"[Milvus] Created partition: {partition_name}")

    def create_vector_index(self):
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        print("[Milvus] Index created on field 'vector'")

    def insert_vector(self, entityID: str, vector: List[float], partition_name: Optional[str] = None):
        if partition_name:
            # 把非法字符全部替换成下划线
            partition_name = re.sub(r'[^0-9a-zA-Z_]', '_', partition_name)
        
        assert self.collection is not None, "Collection not created"
        if self._dim is None:
            self._dim = len(vector)
        if len(vector) != self._dim:
            raise ValueError(f"Vector dim mismatch: expected {self._dim}, got {len(vector)}")

        if partition_name:
            self._ensure_partition(partition_name)

        entities = [[entityID], [vector]]
        # 按 partition 插入（若 partition_name None，则写入 default）
        self.collection.insert(entities, partition_name=partition_name if partition_name else None)
        # self.collection.flush()
        print(f"[Milvus] Inserted (entityID={entityID}) vector_dim={len(vector)}; num_entities={self.collection.num_entities}")

    def count(self) -> int:
        return int(self.collection.num_entities)

    def search(self, vector: List[float], top_k: int = 3, partition_names: Optional[List[str]] = None) -> List[str]:
        self.collection.load(partition_names=partition_names)
        print(f"[Milvus] Searching (dim={len(vector)}) top_k={top_k} partitions={partition_names or 'ALL'}")
        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["entityID"],
            partition_names=partition_names
        )
        # results[0] 为第一个查询向量的 hits
        hits = [hit.entity.get("entityID") for hit in results[0]]
        print(f"[Milvus] Search results: {hits}")
        return hits
