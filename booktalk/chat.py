"""文字问答 CLI（带引用出处）。用法：python -m booktalk.chat"""
from __future__ import annotations

from . import config
from .brain import answer
from .store import BookStore

_MAX_HISTORY_MESSAGES = 8  # 控制 token 用量，避免历史无限增长


def _print_sources(chunks: list[dict]) -> None:
    print("── 来源 ──")
    for i, c in enumerate(chunks, start=1):
        heading = c.get("heading") or "（无标题）"
        page = f" p.{c['page']}" if c.get("page") else ""
        snippet = c["text"].replace("\n", " ")[:120]
        print(f"[{i}] 《{c['source']}》「{heading}」{page}\n    {snippet}…")


def main() -> int:
    if not config.DEEPSEEK_API_KEY:
        print("未配置 DEEPSEEK_API_KEY，请先 cp .env.example .env 并填写 key。")
        return 1

    store = BookStore()
    history: list[dict] = []
    print("booktalk 文字问答（exit 退出）。输入问题：")
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            return 0

        chunks = store.search(q)
        if not chunks:
            print("助手> 没有检索到相关内容，请先运行 python -m booktalk.ingest 导入。")
            continue

        text = answer(q, chunks, history)
        print(f"助手> {text}\n")
        _print_sources(chunks)

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": text})
        history = history[-_MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    raise SystemExit(main())
