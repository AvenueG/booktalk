"""大脑：DeepSeek（OpenAI 兼容接口）做带引用的回答，支持可切换的「人设」。

人设 = 语气 + 示例对话（few-shot），架在不变的引用底线之上：
书里事实要标 [n]，书外常识标「（书外补充）」，绝不把编造的说成书里写的。
"""
from __future__ import annotations

import json
import re

from openai import OpenAI

from . import config

# 复用同一个 OpenAI 客户端（连接池），别每次请求都 new 一个
_client: OpenAI | None = None
_client_key: str | None = None


def _get_client() -> OpenAI:
    global _client, _client_key
    key = config.get_deepseek_key()
    # key 变了就重建客户端（网页设置里改 key 后，下一次请求生效）
    if _client is None or _client_key != key:
        _client = OpenAI(api_key=key, base_url=config.DEEPSEEK_BASE_URL)
        _client_key = key
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


# ---------- 跟着读（主动阅读循环：节奏 + 主动回忆 + 复盘） ----------
_GUIDE_TASKS = {
    "discuss": (
        "用户刚读完下面这一段。请像读书搭子一样，用 2-4 句跟他聊这一段："
        "点出最值得记住的一点，打个比方或给个画面，最后留一个能勾着他继续往下读的钩子或问题。"
        "别复述全文，别一二三条罗列。"
    ),
    "quiz": (
        "用户刚读完下面这一段。请你出一个「主动回忆」问题考他，检查他是不是真抓住了这段的核心。"
        "不要抠冷门细节，要考这段最关键的那个点。只输出问题本身，不要解释、不要给答案、不要编号。"
    ),
    "evaluate": (
        "用户刚读完下面这一段，你之前问了他一个问题，他给出了回答。请评价他的回答："
        "对在哪、漏了什么，语气像朋友一样鼓励，最后用一句话补上这段最关键、他可能漏掉的那个点。"
    ),
    "recap": (
        "用户离开了一会儿，现在回来了。下面是他的阅读进度和最近的对话。"
        "请用最少的话（3-5 句）帮他找回上下文：他读到哪了、刚才在聊什么、接下来该读哪。"
    ),
}


def guide_stream(
    context: str,
    heading: str,
    mode: str,
    question: str = "",
    answer: str = "",
    persona: dict | None = None,
):
    """「跟着读」主动循环的流式回答。

    与 ask 不同：这里上下文是用户正在读的「这一段」本身（或复盘用的进度摘要），
    不走向量检索，直接喂原文。mode：discuss / quiz / evaluate / recap。
    """
    persona = persona or load_persona(config.PERSONA)
    system = _system_prompt(persona) + "\n\n【本次任务】" + _GUIDE_TASKS.get(mode, _GUIDE_TASKS["discuss"])
    parts = []
    if heading and mode != "recap":
        parts.append("【这一段】" + heading)
    parts.append((context or "").strip() or "（无内容）")
    if mode == "evaluate":
        parts.append("【你之前问的问题】" + (question or ""))
        parts.append("【他的回答】" + (answer or "（没回答）"))
    user = "\n\n".join(parts)
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8 if mode in ("quiz", "discuss") else 0.7,
        stream=True,
    )
    for chunk in resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------- 笔记自动整理 ----------
_ORGANIZE_SYSTEM = (
    "你是读书笔记整理助手。把用户零散的读书笔记整理成有条理、能让他日后回看时"
    "「一眼想起当时读了什么、问了什么、为什么这么想」的笔记。\n"
    "要求：\n"
    "1. 按书本章节归位：能对应到某个章节的，归到该章节下；对应不上的放到末尾「零散笔记」。\n"
    "2. 每条尽量保留「问/想法 → 要点 → 原文出处」三个要素，别丢上下文；原文出处用「> 📖」引用。\n"
    "3. 开头写一段极简的「最近在琢磨什么」（2-3 句）+ 一句「接下来读哪」。\n"
    "4. 只整理给定内容，不编造书里没有的；没把握的别硬归类。\n"
    "5. 输出 Markdown：一级标题是书名，二级标题分块，用标题层级而不是一堆列表。"
)


def organize_notes(
    book_title: str,
    notes: str,
    headings: list[str],
    progress_summary: str = "",
) -> str:
    """把零散笔记整理成「按章节归位 + 溯源 + 上下文」的结构化 Markdown（一次性调用）。"""
    toc = "\n".join("- " + h for h in (headings or [])[:80]) or "（无目录）"
    user = (
        f"【书名】{book_title}\n\n"
        f"【目录（前若干章，供归位参考）】\n{toc}\n\n"
        f"【阅读进度】{progress_summary or '（未知）'}\n\n"
        f"【待整理的笔记】\n{notes or '（还没有笔记）'}"
    )
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _ORGANIZE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


# ---------- 双人播客脚本（NotebookLM Audio Overview 式） ----------
_PODCAST_LEN = {
    "short": "约 800 字（5 分钟左右）",
    "medium": "约 1300 字（8 分钟左右）",
    "long": "约 2000 字（12 分钟左右）",
}

# 四种播客格式（对齐 NotebookLM Audio Overview 的 Deep Dive / Brief / Critique / Debate）。
# 共同底线都收在「口语化、只讲提纲、只输出对白」里，格式只改「谁在说、怎么聊」。
# 主播人设与对话手感（所有格式通用，前置拼进 system）：让两位主播像真人朋友闲聊，
# 而不是念稿——这是「像不像 NotebookLM」的关键。
_PODCAST_HOSTS = (
    "【两位主播的人设与对话风格（所有格式通用，务必照做）】\n"
    "小晓：主讲，偏「理论家/讲故事」——看大局、抓主题、爱惊叹爱打比方，先把「最震撼的那个点」抛出来勾人。常用「你想想看」「这事儿特有意思」「我读到这儿都愣了一下」。\n"
    "小宇：追问，偏「务实较真」——补细节、拆逻辑、替听众问「这到底怎么落地/真的吗」，把话题往深里带。口头禅「对，而且……」「等等，这个展开说说」「说白了就是……」「话虽如此，可……」\n"
    "两人角色要能听出差异：小晓负责把事讲得兴奋、有画面；小宇负责较真、质疑、补一句扎心或扎实的细节。\n"
    "对话要像两个真人朋友在聊，不要像在念稿：\n"
    "1. 有来有回、会附和会打断（对、没错、等等、你别说、话虽如此），会接对方的梗、顺着上一句往下说。\n"
    "2. 别用「首先/其次/最后」「接下来我们讲」这种汇报腔；用「还有一点」「更绝的是」「这正好接上……」「你注意到没有」来自然推进。\n"
    "3. 会提炼、会连接：抓一条贯穿始终的主题线来讲故事，把不同章节串起来，而不是逐章复述。\n"
    "4. 会打比方、会给画面、会惊叹、会反问勾人，让听众想听下去。\n"
    "5. 每个主要观点至少配一个提纲里的具体例子（人名/数字/场景），别空谈；小宇要至少一两次「质疑或反着想一下」，让对谈像真有观点碰撞，而不是两个人一致叫好。\n"
    "6. 开头用「冷开场」：第一句直接抛一个最反常识、最抓人的具体细节或判断，别用「今天我们来聊……」这种套话开场。\n"
    "7. 结尾收在「一个最值得带走的点」上，用一句话点透，别用「以上就是今天的全部内容」这种空话收尾。\n"
    "8. 口语短句，别用书面长句、别用 Markdown 符号、别出现 [n] 这类引用标记。\n"
    "9. 只讲提纲里有的内容，不编造。"
)

_PODCAST_FORMATS = {
    "deep": (
        "你是播客脚本写手。根据给定的「提纲」，写一段双主播对谈（Deep Dive）。\n"
        "要求：\n"
        "1. 两个主播，用「小晓：」和「小宇：」区分，一人一段交替着来。\n"
        "2. 小晓是主讲、擅长把事讲生动；小宇爱追问、接话、补画面，两人有来有回、有闲聊感（「对吧」「你瞧」「有意思」）。\n"
        "3. 严格只讲提纲里有的内容，不编造书里没有的细节。\n"
        "4. 口语化、适合朗读：短句、别用 Markdown 列表或符号、别出现 [n] 这类引用标记。\n"
        "5. 开头用「冷开场」直接抛钩子、结尾用「aha 带走的点」收束（见通用要求）。\n"
        "6. 只输出对白，每行一段，形如「小晓：……」。"
    ),
    "brief": (
        "你是播客脚本写手。根据给定的「提纲」，写一段单人快速简报（Brief）。\n"
        "要求：\n"
        "1. 只有一个主播，所有对白都用「小晓：」开头，不要出现第二个主播。\n"
        "2. 两分钟内讲完：只抓最关键的几个要点，别展开细节。\n"
        "3. 严格只讲提纲里有的内容，不编造。\n"
        "4. 口语化、适合朗读：短句、别用 Markdown 符号、别出现 [n] 引用标记。\n"
        "5. 开头一句话点出最值得知道的那一点，结尾一句收束。\n"
        "6. 只输出对白，每行一段，形如「小晓：……」。"
    ),
    "critique": (
        "你是播客脚本写手。根据给定的「提纲」，两位主播对这份材料做批判性点评（Critique）。\n"
        "要求：\n"
        "1. 两个主播，用「小晓：」「小宇：」区分，交替发言。\n"
        "2. 给出建设性、有专家味的反馈：哪里讲得好、论证哪里弱、哪里可以更严谨或更深入，像在帮作者打磨稿子。\n"
        "3. 严格只评提纲里有的内容，不编造材料里没有的。\n"
        "4. 口语化、适合朗读：短句、别用 Markdown 符号、别出现 [n] 引用标记。\n"
        "5. 开头一句点明「我们来点评这份材料」，结尾一两句收束。\n"
        "6. 只输出对白，每行一段，形如「小晓：……」。"
    ),
    "debate": (
        "你是播客脚本写手。根据给定的「提纲」，两位主播围绕材料里的观点展开一场正式辩论（Debate）。\n"
        "要求：\n"
        "1. 两个主播，用「小晓：」「小宇：」区分，轮流陈述。\n"
        "2. 小晓站正方、小宇站反方，各自摆论据、回应对方，观点要有交锋；只基于材料里已有的立场和论据，不编造。\n"
        "3. 口语化、适合朗读：短句、别用 Markdown 符号、别出现 [n] 引用标记。\n"
        "4. 开头一句宣布辩题，结尾一两句收束（不必分出胜负）。\n"
        "5. 只输出对白，每行一段，形如「小晓：……」。"
    ),
}


def podcast_script(
    book_title: str,
    outline: str,
    length: str = "medium",
    format: str = "deep",
    instructions: str = "",
) -> str:
    """把提纲写成播客对白脚本（一次性调用，这是整个播客唯一花 API 的地方）。

    format：deep / brief / critique / debate（对应 NotebookLM 四种格式）。
    instructions：用户的自定义方向，优先级高于格式的默认要求。
    """
    if format == "brief":
        want = "约 350 字（两分钟左右）"
    else:
        want = _PODCAST_LEN.get(length, _PODCAST_LEN["medium"])
    system = _PODCAST_HOSTS + "\n\n" + _PODCAST_FORMATS.get(format, _PODCAST_FORMATS["deep"])
    if instructions.strip():
        system += (
            "\n\n【用户的自定义要求（务必照做，优先级高于上面的默认要求）】\n"
            + instructions.strip()
        )
    user = f"【书名】{book_title}\n\n【提纲】\n{outline}\n\n【时长】{want}，请按这个长度写。"
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
    )
    return resp.choices[0].message.content or ""


def _podcast_answer_system(context: str) -> str:
    """插嘴回答的系统提示；context 是节目刚播到的原文，让主播能精确接回（而非泛泛「接着聊」）。"""
    system = (
        _PODCAST_HOSTS + "\n\n"
        "【本次任务】这是播客播到一半，用户刚插嘴问了个问题。你们两位要像主持人一样接住他：\n"
        "1. 用「小晓：」「小宇：」区分，每轮单独一行，共 2-3 轮、总共 60-120 字，别长篇大论。\n"
        "2. 先随口呼应一下他的问题（一句「好问题」「这问得挺到点子上的」都行），再基于下面文段回答，只讲文段里有的。\n"
        "3. 最后一句要自然地把话题接回刚才的节目，让听众知道节目继续了。"
    )
    if context.strip():
        system += (
            "\n4. 节目刚才正好播到这里：【" + context.strip()
            + "】。你的最后一句要精确接回这里的内容（提到里面具体的人名/事），别只说「接着聊」。"
        )
    return system


def podcast_answer(question: str, chunks: list[dict], context: str = "") -> str:
    """两位主播简短回答用户的「插嘴」，答完自然把话题接回节目（一次性调用）。"""
    book_text = _format_chunks(chunks)
    system = _podcast_answer_system(context)
    user = f"【书本文段】\n{book_text}\n\n【用户插嘴】{question}"
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
    )
    return resp.choices[0].message.content or ""


def _split_turns(text: str) -> list[tuple[str, str]]:
    """把一段「小晓：/小宇：」对白切成一整轮轮 (voice, text)。模型偶尔不按前缀出牌时兜底：整段当一轮。"""
    re_a = re.compile(r"^(?:小晓|晓晓|小A|A|主播A)[：:]\s*(.*)$")
    re_b = re.compile(r"^(?:小宇|宇宇|小B|B|主播B)[：:]\s*(.*)$")
    turns: list[tuple[str, str]] = []
    voice = ""
    cur: list[str] = []

    def flush():
        t = "".join(cur).strip()
        return (voice, t) if (voice and t) else None

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        ma, mb = re_a.match(line), re_b.match(line)
        if ma or mb:
            done = flush()
            if done:
                turns.append(done)
            voice = "小晓：" if ma else "小宇："
            cur = [(ma or mb).group(1)]
        else:
            cur.append(line)
    done = flush()
    if done:
        turns.append(done)
    if not turns:
        turns = [("小晓：", text.strip())]
    return turns


def stream_podcast_answer(question: str, chunks: list[dict], context: str = ""):
    """流式版插嘴回答：把「小晓：/小宇：」对白按「轮」逐条产出，供前端边合成边播放，压首响延迟。

    模型有 0.8 温度、偶尔会不按「小晓：」前缀出牌，导致一轮都切不出来（前端会静音）。
    这里做了兜底：若流式一轮都没产出，退回一次性 `podcast_answer` 再按轮切，保证一定有声音。
    """
    system = _podcast_answer_system(context)
    book_text = _format_chunks(chunks)
    user = f"【书本文段】\n{book_text}\n\n【用户插嘴】{question}"
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        stream=True,
    )
    re_a = re.compile(r"^(?:小晓|晓晓|小A|A|主播A)[：:]\s*(.*)$")
    re_b = re.compile(r"^(?:小宇|宇宇|小B|B|主播B)[：:]\s*(.*)$")
    buf = ""
    voice = ""           # 当前轮主播标签（"小晓：" / "小宇："），空 = 还没开始
    text: list[str] = []
    yielded = 0

    def complete():
        t = "".join(text).strip()
        return (voice, t) if (voice and t) else None

    for chunk in resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        buf += delta
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            ma = re_a.match(line)
            mb = re_b.match(line)
            if ma or mb:
                done = complete()
                if done:
                    yield done
                    yielded += 1
                voice = "小晓：" if ma else "小宇："
                text = [(ma or mb).group(1)]
            else:
                text.append(line)
    done = complete()
    if done:
        yield done
        yielded += 1
    if yielded == 0:
        # 流式一次都没切出「主播：」轮，退回一次性调用兜底（保证有声音，而非静音后直接接回）
        fallback = podcast_answer(question, chunks, context)
        for voice, t in _split_turns(fallback):
            yield voice, t
