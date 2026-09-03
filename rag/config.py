"""RAG 相关配置(集中在一处,方便改)。

三层配置:
  ① embedding:SiliconFlow 的 bge-m3(OpenAI 兼容接口)
  ② 向量库:本地 Qdrant
  ③ 关系库:本地 Postgres(newsrag 库)

重要:本地服务(Qdrant/Postgres)必须绕过 Clash 全局代理,否则 127.0.0.1
会被代理拦截。httpx 用 trust_env=False,psycopg 直连本地不受代理影响。
"""

import os

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
