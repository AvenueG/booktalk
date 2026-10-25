"""集中读取配置与环境变量。"""
from __future__ import annotations

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

# RAG 参数
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
COLLECTION = "book"
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "6"))

# 人设：默认「读书搭子」，可用 --persona 或 /persona 切换
PERSONA = os.getenv("PERSONA", "default")
PERSONAS_DIR = REPO_ROOT / "personas"
