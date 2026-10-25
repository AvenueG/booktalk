"""大脑：DeepSeek（OpenAI 兼容接口）做带引用的回答，支持可切换的「人设」。

人设 = 语气 + 示例对话（few-shot），架在不变的引用底线之上：
书里事实要标 [n]，书外常识标「（书外补充）」，绝不把编造的说成书里写的。
"""
from __future__ import annotations

import json

from openai import OpenAI

from . import config

# 复用同一个 OpenAI 客户端（连接池），别每次请求都 new 一个
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client

# 底线规则：所有「人设」都必须遵守。人设只改变语气和风格，不改变这些底线。
_BASE_RULES = (
    "你是一个读书搭子，基于给定的书本文段 + 你自己的常识，陪用户聊这本书。\n"
    "底线规则（无论你这次的人设是什么，都必须遵守）：\n"
    "1. 用与用户相同的语言回答。\n"
    "2. 书里写的事实 → 标注出处 [n]（n 对应文段编号），但别每句都标。\n"
    "3. 大胆讲活：打比方、场景想象、对比、惊叹、反问、留钩子——这些修辞与想象力不算编造。\n"
    "4. 区分两类内容：书里没写、但属于常识或你本就知道的（比如「木星为什么这么大」）→ 可以讲，但必须标「（书外补充）」，绝不能当成书里的内容标 [n]；连常识都答不上，才说「这个我确实不知道」。\n"
    "5. 顺着用户的情绪走：他说「无聊」「讲好玩点」，就换更生动的讲法，不要拒绝。\n"
    "6. 像人一样说话：有语气、有画面，别处处一二三条。\n"
    "7. 适当用 Markdown 的 **加粗** 突出关键结论、人名、数字，方便快速扫读，但别整段都加粗。"
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
        # 每段截到 700 字再喂给模型：引用出处保留全文（前端弹层另存），
        # 但上下文里只要够回答的「主旨」，省一半输入 token。
        text = (c.get("text") or "").strip()
        if len(text) > 700:
            text = text[:700] + "…"
        lines.append(f"[{i}] 章节「{heading}」{page}：\n{text}")
    return "\n\n".join(lines)


_DEPTH = {
    "brief": "回答尽量简短，一两句话直接给结论，少展开。",
    "deep": "深入展开：多角度、给背景、必要时补书外延伸，但书外内容必须标「（书外补充）」。",
}


def _build_messages(
    question: str,
    chunks: list[dict],
    history: list[dict] | None,
    persona: dict,
    depth: str,
) -> list[dict]:
    system = _system_prompt(persona)
    extra = _DEPTH.get(depth)
    if extra:
        system += f"\n\n【本次回答深度】{extra}"
    context = _format_chunks(chunks)
    user = f"【书本文段】\n{context}\n\n【问题】{question}"
    messages = [{"role": "system", "content": system}]
    if history:
        messages += history[-12:]   # 只带最近 12 条（约 6 轮），省 token、避免超长
    messages.append({"role": "user", "content": user})
    return messages


def answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    persona: dict | None = None,
    depth: str = "standard",
) -> str:
    persona = persona or load_persona(config.PERSONA)
    messages = _build_messages(question, chunks, history, persona, depth)
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL, messages=messages, temperature=0.7
    )
    return resp.choices[0].message.content or ""


def stream_answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    persona: dict | None = None,
    depth: str = "standard",
):
    """流式回答：逐个产出文本增量（OpenAI 兼容 stream 接口），供前端逐字渲染。"""
    persona = persona or load_persona(config.PERSONA)
    messages = _build_messages(question, chunks, history, persona, depth)
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL, messages=messages, temperature=0.7, stream=True
    )
    for chunk in resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def suggest(
    question: str,
    answer: str,
    chunks: list[dict],
    persona: dict | None = None,
    seed: bool = False,
) -> list[str]:
    """生成一组「接着问」的好问题：seed=True 给开场问题，否则给追问。

    横向（相关话题）与纵向（深挖细节）都要有，用于欢迎页和每条回答之后的推荐。
    """
    persona = persona or load_persona(config.PERSONA)
    system = _system_prompt(persona)
    client = _get_client()
    context = _format_chunks(chunks)
    if seed:
        task = (
            "请给出一组「开场问题」：4 个关于这本书、用户可能感兴趣、一句话就能问出的好问题。\n"
            "覆盖：这本书讲了什么 / 最精彩或最意外的部分 / 值得记住的观点 / 可以深挖的人物或事件。"
            "横向（相关话题）与纵向（往下挖细节）都要有。"
        )
        user = f"【书本文段（全书开头）】\n{context}\n\n【任务】{task}"
    else:
        task = (
            "请给出一组「追问」：4 个用户接下来可能感兴趣、值得继续问的好问题。\n"
            "横向（相关话题）与纵向（深挖当前问题的细节）都要有，鼓励用户继续追问。"
        )
        user = (
            f"【书本文段】\n{context}\n\n"
            f"【用户刚问】{question}\n\n"
            f"【你刚才的回答】{answer}\n\n"
            f"【任务】{task}"
        )
    messages = [
        {
            "role": "system",
            "content": system + "\n\n【输出格式】只输出问题本身，一行一个，不要编号、不要引号、不要任何解释。",
        },
        {"role": "user", "content": user},
    ]
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL, messages=messages, temperature=0.8
    )
    text = resp.choices[0].message.content or ""
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.、-—·*•) （( ").strip()
        if line and line not in out:
            out.append(line)
    return out[:4]
