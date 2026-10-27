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
# .env 都已就绪(避免"导入早于 load_dotenv 导致 key 读不到"的时序问题)。
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

# ⑤ M9 Step 2:混合检索 + 重排。
# RAG_MODE:v1 = M3 原链路(定长切块 + dense,旧集合);hybrid = v2 集合 dense+sparse → RRF → rerank。
# 默认值等 Step 3 消融出来再定,先保持 v1,不改变现有 Agent 行为。
RAG_MODE = os.environ.get("RAG_MODE", "v1")
# 8B 而不是 0.6B(M9 §6.4):Hit@5 打平,但 MRR 0.871 vs 0.808,且只有 8B 的分数能设阈值;代价是时延 +1.1s。
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
RECALL_K = int(os.environ.get("RAG_RECALL_K", "30"))           # 每一路召回多少
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "30"))  # RRF 后取多少进 rerank
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))               # 最终给 LLM 几条
# 低于该分的 rerank 结果丢弃。0.5 是 Step 3 在 8B 上标定的:留 98% 相关块、去 70% 不相关块(§6.4)。
# 换回 0.6B 时要设成空(= 不丢):0.6B 的不相关块中位数就有 0.970,这个阈值对它没有意义。
# 注意:阈值只是去噪,**不是拒答**——无答案题仍有 4/10 的 top1 过线,拒答靠 Step 4 的 grade。
_thr = os.environ.get("RAG_RERANK_THRESHOLD", "0.5")
RERANK_THRESHOLD = float(_thr) if _thr else None

# ⑥ M9 Step 4:Agentic RAG 子图(RAG_MODE=agentic)。
RAG_MAX_ROUNDS = int(os.environ.get("RAG_MAX_ROUNDS", "2"))        # 检索总轮数(含首轮)
RAG_FAST_PATH = float(os.environ.get("RAG_FAST_PATH", "0.9"))      # 单需求 + 过滤有效 + top1 ≥ 此值 → 跳过 grade
RAG_DEADLINE_S = float(os.environ.get("RAG_DEADLINE_S", "30"))     # 子图总时限;超时不再改写重检
RAG_WEB_FALLBACK = os.environ.get("RAG_WEB_FALLBACK", "1") == "1"  # 轮数用尽仍缺 → search_news
RAG_HYDE = os.environ.get("RAG_HYDE", "0") == "1"                  # 消融开关,默认关
RAG_EVIDENCE_CHARS = int(os.environ.get("RAG_EVIDENCE_CHARS", "9000"))  # 证据总预算(~3000 token)
