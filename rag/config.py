"""RAG 相关配置(集中在一处,方便改)。

三层配置:
  ① embedding:SiliconFlow 的 bge-m3(OpenAI 兼容接口)
  ② 向量库:本地 Qdrant
  ③ 关系库:本地 Postgres(newsrag 库)

重要:本地服务(Qdrant/Postgres)必须绕过 Clash 全局代理,否则 127.0.0.1
会被代理拦截。httpx 用 trust_env=False,psycopg 直连本地不受代理影响。
"""

import os

from dotenv import load_dotenv

# 在读取任何环境变量之前先加载 .env。放在这里可保证:无论谁先 import 到本模块,
# .env 都已就绪(避免"导入早于 load_dotenv 导致 key 读不到"的时序坑)。
load_dotenv()

# ① Embedding(SiliconFlow bge-m3)
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))  # bge-m3 输出维度是 1024

# ② 向量库(Qdrant)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "finance_corpus")

# ③ 关系库(Postgres)。默认连你本地已有的 newsrag 库。
PG_DSN = os.environ.get(
    "PG_DSN",
    "host=127.0.0.1 port=5432 dbname=newsrag user=postgres password=postgres",
)

# ④ M9:v2 语料(父子块 + 元数据 + dense/sparse 双向量)。
# 旧集合 finance_corpus 是匿名单向量,加不了 sparse,保留作消融对照(M9 §4.2 第 0 行)。
QDRANT_COLLECTION_V2 = os.environ.get("QDRANT_COLLECTION_V2", "finance_corpus_v2")
CORPUS_RAW_DIR = os.environ.get("CORPUS_RAW_DIR", "data/corpus/raw")
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "32"))
