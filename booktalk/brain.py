"""大脑：DeepSeek（OpenAI 兼容接口）做带引用的回答，支持可切换的「人设」。

人设 = 语气 + 示例对话（few-shot），架在不变的引用底线之上：
书里事实要标 [n]，书外常识标「（书外补充）」，绝不把编造的说成书里写的。
"""
from __future__ import annotations

import json

from openai import OpenAI

from . import config

# 底线规则：所有「人设」都必须遵守。人设只改变语气和风格，不改变这些底线。
_BASE_RULES = (
    "你是一个读书搭子，基于给定的书本文段 + 你自己的常识，陪用户聊这本书。\n"
    "底线规则（无论你这次的人设是什么，都必须遵守）：\n"
    "1. 用与用户相同的语言回答。\n"
    "2. 书里写的事实 → 标注出处 [n]（n 对应文段编号），但别每句都标。\n"
    "3. 大胆讲活：打比方、场景想象、对比、惊叹、反问、留钩子——这些修辞与想象力不算编造。\n"
    "4. 区分两类内容：书里没写、但属于常识或你本就知道的（比如「木星为什么这么大」）→ 可以讲，但必须标「（书外补充）」，绝不能当成书里的内容标 [n]；连常识都答不上，才说「这个我确实不知道」。\n"
    "5. 顺着用户的情绪走：他说「无聊」「讲好玩点」，就换更生动的讲法，不要拒绝。\n"
    "6. 像人一样说话：有语气、有画面，别处处一二三条。"
)


def load_persona(name: str) -> dict:
    """从 personas/ 目录加载人设；找不到就退回 default。"""
    path = config.PERSONAS_DIR / f"{name}.json"
    if not path.exists():
        path = config.PERSONAS_DIR / "default.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"name": "default", "label": "读书搭子", "tone": "", "example": []}


def list_personas() -> list[dict]:
    """列出所有人设（按文件名排序）。"""
    out = []
    for p in sorted(config.PERSONAS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def _system_prompt(persona: dict) -> str:
    parts = [_BASE_RULES]
    tone = (persona.get("tone") or "").strip()
    if tone:
        label = persona.get("label") or persona.get("name") or ""
        parts.append(f"【你这次的人设】{label}。{tone}")
    example = persona.get("example") or []
    if example:
        lines = []
        for turn in example:
            lines.append(f"用户：{turn.get('user', '')}")
            lines.append(f"你：{turn.get('assistant', '')}")
        parts.append("【示例语气（模仿这种感觉，别照抄里面的内容）】\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _format_chunks(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        heading = c.get("heading") or "（无标题）"
        page = f" 第{c['page']}页" if c.get("page") else ""
        lines.append(f"[{i}] 章节「{heading}」{page}：\n{c['text']}")
    return "\n\n".join(lines)


def answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    persona: dict | None = None,
) -> str:
    persona = persona or load_persona(config.PERSONA)
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    context = _format_chunks(chunks)
    user = f"【书本文段】\n{context}\n\n【问题】{question}"
    messages = [{"role": "system", "content": _system_prompt(persona)}]
    messages += history or []
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL, messages=messages, temperature=0.7
    )
    return resp.choices[0].message.content or ""
