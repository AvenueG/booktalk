"""检索冒烟测试：对正在运行的服务打 N 个问题，校验「有来源 + 有回答 + 带引用 + 流式收尾」。

零新依赖（只用标准库）。先启动服务再跑：
    python scripts/smoke.py
"""
from __future__ import annotations

import json
import re
import urllib.request

BASE = "http://127.0.0.1:8000"

# 黄金问题：覆盖不同主题，用来守住检索质量（书换了要换这组）
QUESTIONS = [
    "马斯克的童年是怎么度过的？",
    "特斯拉和 SpaceX 的关键转折点是什么？",
    "马斯克的管理风格有什么特点？",
    "书里写了哪些失败和翻车？",
    "他和家人、合伙人的关系怎么样？",
    "马斯克为什么要去火星？",
]

CITE = re.compile(r"\[\d+\]")


def ask(q: str) -> tuple[list, str, bool]:
    body = json.dumps(
        {"question": q, "persona": "default", "depth": "standard", "history": []}
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/ask/stream", data=body, headers={"Content-Type": "application/json"}
    )
    sources, text, done = [], "", False
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") == "sources":
                sources = ev.get("sources", [])
            elif ev.get("type") == "delta":
                text += ev.get("text", "")
            elif ev.get("type") == "done":
                done = True
    return sources, text, done


def main() -> int:
    ok = 0
    print(f"对 {BASE} 做检索冒烟，共 {len(QUESTIONS)} 问\n")
    for i, q in enumerate(QUESTIONS, 1):
        try:
            sources, text, done = ask(q)
            good = bool(sources) and len(text) >= 20 and done
            ok += 1 if good else 0
            head = (sources[0].get("heading") if sources else "（无）")
            print(
                f"[{i}] {'OK ' if good else 'FAIL'} 来源{len(sources)} 答案{len(text)}字 "
                f"引用{'有' if CITE.search(text) else '无'} · {q}\n      首个来源：{head}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] FAIL 异常 {e} · {q}")
    print(f"\n{ok}/{len(QUESTIONS)} 通过")
    return 0 if ok == len(QUESTIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
