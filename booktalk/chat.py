"""文字问答 CLI（带引用出处 + 可切换人设）。

用法：
    python -m booktalk.chat                # 默认人设「读书搭子」
    python -m booktalk.chat --persona romantic
聊天内命令：/persona 查看、/persona <名字> 切换、/help、exit。
"""
from __future__ import annotations

import sys

from . import config
from .brain import answer, list_personas, load_persona
from .store import BookStore

_MAX_HISTORY_MESSAGES = 8  # 控制 token 用量，避免历史无限增长


def _print_sources(chunks: list[dict]) -> None:
    print("── 来源 ──")
    for i, c in enumerate(chunks, start=1):
        heading = c.get("heading") or "（无标题）"
        page = f" p.{c['page']}" if c.get("page") else ""
        snippet = c["text"].replace("\n", " ")[:120]
        print(f"[{i}] 《{c['source']}》「{heading}」{page}\n    {snippet}…")


def _pick_persona(argv: list[str]) -> str:
    """从命令行参数里取人设名：--persona=名 或 --persona 名 / -p 名。"""
    name = config.PERSONA
    for i, a in enumerate(argv):
        if a.startswith("--persona="):
            name = a.split("=", 1)[1]
        elif a in ("--persona", "-p") and i + 1 < len(argv):
            name = argv[i + 1]
    return name


def main() -> int:
    if not config.DEEPSEEK_API_KEY:
        print("未配置 DEEPSEEK_API_KEY，请先 cp .env.example .env 并填写 key。")
        return 1

    store = BookStore()
    persona = load_persona(_pick_persona(sys.argv[1:]))
    history: list[dict] = []

    print("booktalk 文字问答（exit 退出，/persona 切换人设，/help 帮助）。")
    print(f"当前人设：{persona.get('label') or persona.get('name')}")
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
        if q == "/help":
            names = "、".join(p.get("name", "?") for p in list_personas())
            print(f"命令：/persona 查看人设；/persona <名字> 切换（可选：{names}）；exit 退出。")
            continue
        if q == "/persona":
            items = "、".join(
                f"{p.get('name')}（{p.get('label', '')}）" for p in list_personas()
            )
            print(f"当前人设：{persona.get('label') or persona.get('name')}。可选：{items}")
            continue
        if q.startswith("/persona "):
            persona = load_persona(q[len("/persona ") :].strip())
            history = []  # 换人设后清掉旧语气的历史，避免串味
            print(f"已切换人设：{persona.get('label') or persona.get('name')}")
            continue

        chunks = store.search(q)
        if not chunks:
            print("助手> 没有检索到相关内容，请先运行 python -m booktalk.ingest 导入。")
            continue

        text = answer(q, chunks, history, persona)
        print(f"助手> {text}\n")
        _print_sources(chunks)

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": text})
        history = history[-_MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    raise SystemExit(main())
