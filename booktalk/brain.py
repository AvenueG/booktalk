"""大脑：DeepSeek（OpenAI 兼容接口）做带引用的回答。"""
from __future__ import annotations

from openai import OpenAI

from . import config

_SYSTEM = (
    "你是一个严谨的读书助手，基于用户提供的书本文段回答问题。\n"
    "规则：\n"
    "1. 用与用户问题相同的语言回答。\n"
    "2. 只依据给出的文段，在句末用 [n] 标注引用编号（n 对应文段编号）。\n"
    "3. 文段不足以回答时，直接说「书里没有相关内容」，不要编造。\n"
    "4. 简洁、有条理，按需分点。"
)


def _format_chunks(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        heading = c.get("heading") or "（无标题）"
        page = f" 第{c['page']}页" if c.get("page") else ""
        lines.append(f"[{i}] 章节「{heading}」{page}：\n{c['text']}")
    return "\n\n".join(lines)


def answer(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    context = _format_chunks(chunks)
    user = f"【书本文段】\n{context}\n\n【问题】{question}"
    messages = [{"role": "system", "content": _SYSTEM}]
    messages += history or []
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL, messages=messages, temperature=0.3
    )
    return resp.choices[0].message.content or ""
