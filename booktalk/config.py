"""集中读取配置与环境变量。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# 仓库根目录：booktalk/config.py 往上两级
REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

# 部分地区连不上 huggingface.co，默认走镜像（可用 .env 里的 HF_ENDPOINT 覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

BOOKS_DIR = REPO_ROOT / "data" / "books"
INDEX_DIR = REPO_ROOT / "data" / "index"
MARKDOWN_DIR = REPO_ROOT / "data" / "markdown"
NOTES_DIR = REPO_ROOT / "data" / "notes"
SESSION_DIR = REPO_ROOT / "data" / "session"
KEYPOINTS_DIR = REPO_ROOT / "data" / "keypoints"
PODCAST_DIR = REPO_ROOT / "data" / "podcast"   # 播客脚本缓存 + 已保存的 mp3

# RAG 参数
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# 模型跑在哪个设备上。默认 cpu：Apple Silicon 上用 MPS 会偶发 Metal 崩溃（
# IOGPUMetalCommandBuffer 断言）把整个服务打挂。追求速度可设 MODEL_DEVICE=mps，但要接受这个风险。
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
COLLECTION = "book"
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "6"))
# 重排器（bge-reranker-v2-m3）在 CPU 上逐条算很慢：候选越多、文本越长越卡。
# 只对最靠前的这些条做重排、每条只喂前 RERANK_MAX_CHARS 字，足够覆盖 TOP_K_FINAL，
# 又能把「插嘴」首响从 20+ 秒压到几秒。追求更准可调大，追求更快可调小。
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "400"))

# 人设：默认「读书搭子」，可用 --persona 或 /persona 切换
PERSONA = os.getenv("PERSONA", "default")
PERSONAS_DIR = REPO_ROOT / "personas"


# ---------- 运行时 API key（网页「设置」面板里填，优先级高于 .env）----------
# 单机单用户，不设鉴权；key 明文存 data/config.json（data/ 已 gitignore），前端密码框不回显。
_DATA_DIR = REPO_ROOT / "data"
_CONFIG_FILE = _DATA_DIR / "config.json"
_runtime_api_key: str | None = None


def _load_runtime_config() -> None:
    """启动时读 data/config.json，把用户之前在网页里填的 key 加载进内存。"""
    global _runtime_api_key
    try:
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            if data.get("api_key"):
                _runtime_api_key = data["api_key"]
    except Exception:  # noqa: BLE001
        pass


def get_deepseek_key() -> str:
    """返回运行时覆盖的 key，没有则回退 .env 的 DEEPSEEK_API_KEY。"""
    return _runtime_api_key or DEEPSEEK_API_KEY


def set_api_key(key: str) -> None:
    """网页设置里保存 API key：写内存 + 持久化到 data/config.json。"""
    global _runtime_api_key
    _runtime_api_key = key.strip() or None
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(
        json.dumps({"api_key": _runtime_api_key}, ensure_ascii=False),
        encoding="utf-8",
    )


_load_runtime_config()
