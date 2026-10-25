"""本地网页版：FastAPI 后端 + 单页前端。

用法：
    python -m booktalk.web
    然后浏览器打开 http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import config
from .brain import answer, guide_stream, list_personas, load_persona, organize_notes, podcast_answer, podcast_script, stream_answer, stream_podcast_answer, suggest
from .store import BookStore

app = FastAPI(title="booktalk")

_store: BookStore | None = None

_OLD_SESSION_PATH = config.NOTES_DIR.parent / "session.json"


def get_store() -> BookStore:
    global _store
    if _store is None:
        _store = BookStore()  # 首次加载 bge-m3 / 重排模型，约十几秒
        _migrate_legacy(_store)
    return _store


def _books() -> list[str]:
    """已导入的书（markdown 文件名 stem，按名排序）。"""
    if not config.MARKDOWN_DIR.exists():
        return []
    return sorted(f.stem for f in config.MARKDOWN_DIR.glob("*.md"))


def _first_book() -> str:
    names = _books()
    return names[0] if names else ""


def _book_markdown(name: str) -> str:
    path = config.MARKDOWN_DIR / f"{name}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="没有这本书")
    return path.read_text(encoding="utf-8")


def _headings(name: str) -> list[str]:
    """书的章节标题（Markdown 的一到六级标题），供笔记整理时归位参考。"""
    out = []
    for line in _book_markdown(name).split("\n"):
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _notes_path(book: str) -> Path:
    return config.NOTES_DIR / f"{book}.md"


def _session_path(book: str) -> Path:
    return config.SESSION_DIR / f"{book}.json"


def _migrate_legacy(store: BookStore) -> None:
    """历史数据迁移：旧索引没有 book 字段，旧会话是全局 session.json。"""
    try:
        names = _books()
        if not names:
            return
        # 只有一本书时，把缺 book 字段的旧块补成这本书（避免强制重导）
        if len(names) == 1:
            n = store.tag_legacy_points(names[0])
            if n:
                print(f"[booktalk] 已给 {n} 个旧块补上 book={names[0]}")
        # 旧全局 session.json → 第一本书的会话
        if _OLD_SESSION_PATH.exists():
            config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
            target = _session_path(names[0])
            if not target.exists():
                shutil.move(str(_OLD_SESSION_PATH), str(target))
    except Exception:  # noqa: BLE001
        pass


# ---------- 请求模型 ----------
class AskRequest(BaseModel):
    question: str
    persona: str = "default"
    depth: str = "standard"
    history: list[dict] = []
    book: str = ""
    context: str = ""   # 显式给一段原文时（如「跟着读·聊聊」）跳过检索，直接用这段
    heading: str = ""


class ContinueRequest(BaseModel):
    history: list[dict] = []
    persona: str = "default"
    depth: str = "standard"
    book: str = ""


class TTSRequest(BaseModel):
    text: str
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "-8%"


class SuggestRequest(BaseModel):
    question: str = ""
    answer: str = ""
    persona: str = "default"
    seed: bool = False
    book: str = ""


class NotesRequest(BaseModel):
    content: str


class OrganizeRequest(BaseModel):
    content: str = ""
    progress: str = ""


class PodcastRequest(BaseModel):
    length: str = "medium"       # short | medium | long
    format: str = "deep"         # deep | brief | critique | debate
    chapters: list[str] = []     # 选中的章节标题；空 = 全书
    instructions: str = ""       # 自定义方向（可选）


class PodcastAudioRequest(BaseModel):
    script: str = ""
    save: bool = False           # True = 落盘保存为一条可回放的播客，返回 {ok,id,url}
    format: str = "deep"
    length: str = "medium"
    chapters: list[str] = []
    instructions: str = ""


class PodcastAnswerRequest(BaseModel):
    question: str = ""
    context: str = ""


class SessionRequest(BaseModel):
    messages: list[dict] = []


class GuideRequest(BaseModel):
    context: str = ""
    heading: str = ""
    mode: str = "discuss"   # discuss | quiz | evaluate | recap
    question: str = ""
    answer: str = ""
    persona: str = "default"
    book: str = ""


class SettingsRequest(BaseModel):
    api_key: str = ""


# ---------- 页面 ----------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    path = Path(__file__).parent / "static" / "index.html"
    return path.read_text(encoding="utf-8")


@app.get("/api/personas")
def personas() -> list[dict]:
    return list_personas()


# ---------- 设置 ----------
@app.get("/api/settings")
def get_settings() -> dict:
    """返回当前配置状态；不回传明文 key，只给 has_key。"""
    return {
        "has_key": bool(config.get_deepseek_key()),
        "model": config.DEEPSEEK_MODEL,
        "base_url": config.DEEPSEEK_BASE_URL,
    }


@app.post("/api/settings")
def save_settings(req: SettingsRequest) -> dict:
    """网页「设置」面板里保存 API key（覆盖 .env，持久化到 data/config.json）。"""
    key = (req.api_key or "").strip()
    if not key:
        return {"ok": False, "error": "API key 不能为空"}
    config.set_api_key(key)
    return {"ok": True, "has_key": True}


# ---------- 书 ----------
@app.get("/api/books")
def books() -> list[dict]:
    return [{"name": b} for b in _books()]


@app.get("/api/book/{name}")
def book(name: str) -> dict:
    return {"name": name, "markdown": _book_markdown(name)}


@app.get("/api/book/{name}/headings")
def book_headings(name: str) -> dict:
    """某本书的章节标题（轻量，供播客「选篇章」等下拉用）。"""
    return {"headings": _headings(name)}


@app.post("/api/import")
def import_upload(file: UploadFile = File(...)) -> dict:
    """上传一本书并导入（同步长请求：Docling 解析 + bge-m3 嵌入可能 1-2 分钟）。"""
    from .ingest import import_book  # 延迟导入，避免启动就拉 docling

    store = get_store()  # 复用运行中的 store，避免 Qdrant 单进程锁冲突
    filename = file.filename or "book"
    book_name = Path(filename).stem or "未命名"
    tmp = config.BOOKS_DIR / ("_upload_" + filename)
    config.BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # 已存在的书：先清掉它的旧索引，再重建（换同一本书的更新版本）
        if book_name in _books():
            store.clear(book_name)
            # markdown 变了，旧中心句缓存作废
            key_cache = config.KEYPOINTS_DIR / f"{book_name}.json"
            if key_cache.exists():
                try:
                    key_cache.unlink()
                except Exception:  # noqa: BLE001
                    pass
            # 播客脚本缓存也作废（书内容变了）
            pod_cache = config.PODCAST_DIR / f"{book_name}.json"
            if pod_cache.exists():
                try:
                    pod_cache.unlink()
                except Exception:  # noqa: BLE001
                    pass
        n = import_book(tmp, book=book_name, store=store)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"ok": True, "name": book_name, "chunks": n}


# ---------- 问答 ----------
def _ask_chunks(req: AskRequest, store: BookStore) -> list[dict]:
    """有显式 context 就用它（不检索）；否则按问题检索。"""
    if (req.context or "").strip():
        return [{"source": req.book, "heading": req.heading or "（当前段落）", "text": req.context, "page": ""}]
    return store.search(req.question, book=req.book or None)


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    store = get_store()
    chunks = _ask_chunks(req, store)
    if not chunks:
        return {"answer": "没有检索到相关内容，先确认书已经导入。", "sources": []}
    persona = load_persona(req.persona)
    text = answer(req.question, chunks, persona=persona, depth=req.depth)
    sources = [
        {
            "source": c.get("source"),
            "heading": c.get("heading") or "（无标题）",
            "page": c.get("page"),
            "text": c.get("text"),
        }
        for c in chunks
    ]
    return {"answer": text, "sources": sources}


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    store = get_store()
    chunks = _ask_chunks(req, store)
    sources = [
        {
            "source": c.get("source"),
            "heading": c.get("heading") or "（无标题）",
            "page": c.get("page"),
            "text": c.get("text"),
        }
        for c in chunks
    ]
    persona = load_persona(req.persona) if chunks else None

    def gen():
        yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False) + "\n"
        if not chunks:
            yield json.dumps({"type": "delta", "text": "没有检索到相关内容，先确认书已经导入。"}, ensure_ascii=False) + "\n"
        else:
            try:
                for delta in stream_answer(req.question, chunks, history=req.history, persona=persona, depth=req.depth):
                    yield json.dumps({"type": "delta", "text": delta}, ensure_ascii=False) + "\n"
            except Exception as e:
                yield json.dumps({"type": "delta", "text": f"\n\n（出错了：{e}）"}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/continue")
def continue_talk(req: ContinueRequest) -> dict:
    # 播客续讲：顺着上一轮自然往下聊（不提问、不等用户），给 2-4 句。
    query = ""
    for m in reversed(req.history):
        if m.get("role") in ("user", "assistant"):
            query = m.get("content", "")
            break
    if query:
        chunks = get_store().search(query, book=req.book or None)
        instruction = (
            "你正在以播客主播的方式陪用户聊这本书。顺着刚才的话自然往下讲："
            "可以展开一个细节、补一个生动的例子，或自然过渡到相邻话题。"
            "像播客一样自说自话、有语气、有画面感；不要向用户提问、不要等用户回应。"
            "控制在 2-4 句，用与用户相同的语言。"
        )
    else:
        # 开场：还没有对话时，用全书开头给播客起个头（点「播客」就要真的开始讲）
        book_name = req.book or _first_book()
        intro = _book_markdown(book_name)[:4000] if book_name else ""
        chunks = [{"source": book_name, "heading": "（全书开头）", "text": intro, "page": ""}] if intro else []
        instruction = (
            "用户刚开始听这本书的播客。请用播客主播的方式开场："
            "用 3-5 句自然、口语、有画面感地介绍这本书和它最抓人的一个点，先勾住听众。"
            "不要向用户提问、不要等回应；讲完自然停在这里，等用户插嘴。用与用户相同的语言。"
        )
    if not chunks:
        return {"text": ""}
    persona = load_persona(req.persona)
    text = answer(instruction, chunks, history=req.history, persona=persona, depth=req.depth)
    return {"text": text}


# ---------- TTS ----------
@app.post("/api/tts")
async def tts(req: TTSRequest) -> Response:
    # edge-tts：微软神经语音（免费、联网），音色自然、断句正确；替代浏览器自带 TTS。
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 edge-tts（pip install edge-tts）")
    text = (req.text or "").strip()
    if not text:
        return Response(content=b"", media_type="audio/mpeg")
    try:
        communicate = edge_tts.Communicate(text, voice=req.voice, rate=req.rate)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return Response(content=bytes(buf), media_type="audio/mpeg")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TTS 失败：{e}")


# ---------- 跟着读（主动阅读循环：节奏 / 主动回忆 / 复盘） ----------
@app.post("/api/guide")
def guide_route(req: GuideRequest) -> StreamingResponse:
    persona = load_persona(req.persona)

    def gen():
        try:
            for delta in guide_stream(
                req.context, req.heading, req.mode, req.question, req.answer, persona=persona
            ):
                yield json.dumps({"type": "delta", "text": delta}, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"type": "delta", "text": f"\n\n（出错了：{e}）"}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------- 推荐问题 ----------
@app.post("/api/suggest")
def suggest_questions(req: SuggestRequest) -> dict:
    persona = load_persona(req.persona)
    if req.seed:
        book_name = req.book or _first_book()
        intro = _book_markdown(book_name)[:4000] if book_name else ""
        chunks = [
            {"source": book_name, "heading": "（全书开头）", "text": intro, "page": ""}
        ]
    else:
        chunks = get_store().search(req.question, book=req.book or None)[:3]
    if not chunks:
        return {"questions": []}
    qs = suggest(req.question, req.answer, chunks, persona=persona, seed=req.seed)
    return {"questions": qs}


# ---------- 中心句提炼（抽取式：本地 jieba 打分，不耗 API） ----------
_SENT_SPLIT = re.compile(r"([。！？；!?;])")


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句，保留标点（与前端 splitSentences 一一对应）。"""
    parts = _SENT_SPLIT.split(text)
    out = []
    for i in range(0, len(parts) - 1, 2):
        out.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1 and parts[-1].strip():
        out.append(parts[-1])
    return [s for s in out if s.strip()]


def _key_indices(paragraph: str) -> list[int]:
    """返回段落里「中心句」的句子下标（0-based）。

    打分用词频（TF）：每句 = 句中出现的词在整段里的出现次数之和，
    再除以句长，避免长句天然占优。取最高 1 句；长段落（>=5 句）取前 2 句。
    """
    import jieba  # 延迟导入，省启动时间

    sentences = _split_sentences(paragraph)
    if len(sentences) <= 1:
        return []
    tokens = [[w for w in jieba.cut(s) if len(w.strip()) >= 2] for s in sentences]
    freq: dict[str, int] = {}
    for ws in tokens:
        for w in ws:
            freq[w] = freq.get(w, 0) + 1
    scored = []
    for i, ws in enumerate(tokens):
        s = sum(freq[w] for w in ws)
        scored.append((s / max(len(sentences[i]), 1), i))
    scored.sort(key=lambda x: -x[0])
    n = 2 if len(sentences) >= 5 else 1
    return sorted(i for _, i in scored[:n])


@app.get("/api/keypoints")
def keypoints(book: str = "") -> dict:
    book = book or _first_book()
    if not book:
        return {"paragraphs": []}
    try:
        import jieba  # noqa: F401  确认已安装，_key_indices 里还会再 import（缓存，无开销）
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 jieba（pip install jieba）")

    cache_path = config.KEYPOINTS_DIR / f"{book}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    markdown = _book_markdown(book)
    paragraphs = []
    for line in markdown.split("\n"):
        # 与前端 renderBook 完全一致：标题行/空行不算段落
        if re.match(r"^(#{1,6})\s+(.*)$", line) or not line.strip():
            continue
        paragraphs.append({"key": _key_indices(line.strip())})

    result = {"paragraphs": paragraphs}
    try:
        config.KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return result


# ---------- 笔记 ----------
@app.get("/api/notes")
def notes(book: str = "") -> dict:
    book = book or _first_book()
    if not book:
        return {"name": "", "content": ""}
    path = _notes_path(book)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"name": book, "content": content}


@app.post("/api/notes")
def save_notes(req: NotesRequest, book: str = "") -> dict:
    book = book or _first_book()
    if not book:
        return {"ok": False, "error": "还没有导入书"}
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    _notes_path(book).write_text(req.content, encoding="utf-8")
    return {"ok": True, "path": str(_notes_path(book))}


@app.post("/api/notes/organize")
def organize(req: OrganizeRequest, book: str = "") -> dict:
    """一键整理：按章节归位 + 溯源 + 保留上下文，返回整理后的 Markdown。"""
    book = book or _first_book()
    if not book:
        return {"ok": False, "error": "还没有导入书"}
    try:
        content = organize_notes(book, req.content or "", _headings(book), req.progress or "")
        return {"ok": True, "content": content}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ---------- 双人播客（NotebookLM Audio Overview 式：读全书免费 → 一次生成脚本 → 双音色合成） ----------
_PODCAST_VOICE_A = "zh-CN-XiaoxiaoNeural"   # 小晓（女声）
_PODCAST_VOICE_B = "zh-CN-YunxiNeural"      # 小宇（男声）


def _book_outline(book: str, chapters: list[str] | None = None, max_chars: int = 6000) -> str:
    """零成本「提纲」：章节标题 + 每段中心句（jieba 抽取式，不调 LLM）。

    chapters 给定时只收录这些章节（标题精确匹配）；其下的子标题/段落一并收进来，
    直到遇到「同级或更高级、且不在选中列表」的标题为止。空/None = 全书。
    """
    want = set(chapters) if chapters else None
    active = want is None
    active_level = 0
    parts = []
    for line in _book_markdown(book).split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if want is not None:
                if title in want:
                    active = True
                    active_level = level
                    parts.append("\n## " + title)
                elif active and level > active_level:
                    # 选中章节下的子标题，照收
                    parts.append("\n## " + title)
                else:
                    active = False
            else:
                parts.append("\n## " + title)
            continue
        if not line.strip():
            continue
        if not active:
            continue
        para = line.strip()
        sents = _split_sentences(para)
        picked = [sents[i] for i in _key_indices(para) if i < len(sents)]
        if picked:
            parts.append("· " + " ".join(picked))
    outline = "\n".join(parts)
    if len(outline) > max_chars:
        outline = outline[:max_chars] + "\n…（后面章节略）"
    return outline


def _parse_turns(script: str) -> list[tuple[str, str]]:
    """把「小晓：…/小宇：…」脚本拆成 [(voice, text), ...]；没标记的行归到上一人。"""
    turns = []
    cur_voice = _PODCAST_VOICE_A
    cur = []
    for raw in (script or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        a = re.match(r"^(?:小晓|晓晓|小A|A|主播A)[：:]\s*(.*)$", line)
        if a:
            if cur:
                turns.append((cur_voice, "".join(cur)))
                cur = []
            cur_voice = _PODCAST_VOICE_A
            cur.append(a.group(1))
            continue
        b = re.match(r"^(?:小宇|宇宇|小B|B|主播B)[：:]\s*(.*)$", line)
        if b:
            if cur:
                turns.append((cur_voice, "".join(cur)))
                cur = []
            cur_voice = _PODCAST_VOICE_B
            cur.append(b.group(1))
            continue
        cur.append(line)
    if cur:
        turns.append((cur_voice, "".join(cur)))
    out = []
    for v, t in turns:
        t = t.strip()
        if not t:
            continue
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + t)
        else:
            out.append((v, t))
    return out


def _podcast_key(length: str, fmt: str, chapters: list[str], instructions: str) -> str:
    """把一组生成参数折成稳定短哈希，作为脚本缓存的键（换参数=换缓存，不串）。"""
    import hashlib

    blob = json.dumps(
        {
            "length": length,
            "format": fmt,
            "chapters": sorted(chapters),
            "instructions": instructions,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


@app.post("/api/podcast")
def podcast_route(req: PodcastRequest, book: str = "") -> dict:
    """生成播客脚本（唯一花 API 的步骤；脚本按书+参数缓存，重放不花钱）。"""
    book = book or _first_book()
    if not book:
        return {"ok": False, "error": "还没有导入书"}
    length = req.length if req.length in ("short", "medium", "long") else "medium"
    fmt = req.format if req.format in ("deep", "brief", "critique", "debate") else "deep"
    chapters = [c.strip() for c in (req.chapters or []) if c.strip()]
    instructions = (req.instructions or "").strip()
    key = _podcast_key(length, fmt, chapters, instructions)
    cache_dir = config.PODCAST_DIR
    cache_path = cache_dir / f"{book}.json"
    cache = {}
    try:
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache.get(key, {}).get("script"):
                return {"ok": True, "script": cache[key]["script"], "cached": True}
    except Exception:  # noqa: BLE001
        cache = {}
    try:
        outline = _book_outline(book, chapters or None)
        script = podcast_script(book, outline, length, format=fmt, instructions=instructions)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache[key] = {"script": script}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "script": script, "cached": False}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _podcasts_dir() -> Path:
    d = config.PODCAST_DIR / "saved"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _podcast_title(book: str, fmt: str, chapters: list[str]) -> str:
    label = {"deep": "深度对谈", "brief": "简报", "critique": "批判点评", "debate": "辩论"}.get(fmt, "播客")
    if chapters:
        return f"{book} · {label} · {len(chapters)}章"
    return f"{book} · {label}"


@app.post("/api/podcast/audio", response_model=None)
async def podcast_audio(req: PodcastAudioRequest, book: str = "") -> Response | dict:
    """双音色合成整段音频（edge-tts 免费；并发合成各句再按序拼接）。

    save=True 时落盘保存为一条可回放的播客，返回 {ok, id, url}；否则返回原始 mp3（插嘴临时播放）。
    """
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 edge-tts（pip install edge-tts）")
    turns = _parse_turns(req.script)
    if not turns:
        return Response(content=b"", media_type="audio/mpeg")

    async def synth(i: int, voice: str, text: str) -> tuple[int, bytes]:
        communicate = edge_tts.Communicate(text, voice=voice, rate="-6%")
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return i, bytes(buf)

    try:
        results = await asyncio.gather(*[synth(i, v, t) for i, (v, t) in enumerate(turns)])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"音频合成失败：{e}")
    results.sort(key=lambda x: x[0])
    audio = b"".join(b for _, b in results)
    if not req.save:
        return Response(content=audio, media_type="audio/mpeg")
    book = book or _first_book()
    if not book:
        return {"ok": False, "error": "还没有导入书"}
    import time
    import uuid

    pid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    meta = {
        "id": pid,
        "book": book,
        "title": _podcast_title(book, req.format, req.chapters or []),
        "format": req.format,
        "length": req.length,
        "chapters": req.chapters or [],
        "instructions": req.instructions or "",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "size": len(audio),
        "script": req.script,
    }
    (_podcasts_dir() / f"{pid}.mp3").write_bytes(audio)
    (_podcasts_dir() / f"{pid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "id": pid, "url": f"/api/podcast/{pid}.mp3", "title": meta["title"]}


@app.get("/api/podcasts")
def podcasts(book: str = "") -> dict:
    """已保存的播客列表（按书筛；按创建时间倒序）。"""
    out = []
    for p in _podcasts_dir().glob("*.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if book and m.get("book") != book:
            continue
        out.append({k: m.get(k) for k in ("id", "book", "title", "format", "length", "created", "size", "script")})
    out.sort(key=lambda x: x.get("created", ""), reverse=True)
    return {"items": out}


@app.get("/api/podcast/{pid}.mp3")
def podcast_mp3(pid: str) -> FileResponse:
    p = _podcasts_dir() / f"{pid}.mp3"
    if not p.exists():
        raise HTTPException(status_code=404, detail="没有这条播客")
    return FileResponse(p, media_type="audio/mpeg")


@app.delete("/api/podcast/{pid}")
def podcast_delete(pid: str) -> dict:
    for p in (_podcasts_dir() / f"{pid}.mp3", _podcasts_dir() / f"{pid}.json"):
        if p.exists():
            p.unlink()
    return {"ok": True}


@app.post("/api/podcast/answer")
def podcast_answer_route(req: PodcastAnswerRequest, book: str = "") -> dict:
    """插嘴：两位主播针对用户的问题，基于书里内容给一段 2-3 轮对白。"""
    book = book or _first_book()
    if not book:
        return {"ok": False, "error": "还没有导入书"}
    chunks = get_store().search(req.question, book=book or None)
    if not chunks:
        return {"ok": False, "error": "没检索到相关内容"}
    try:
        script = podcast_answer(req.question, chunks, req.context or "")
        return {"ok": True, "script": script}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.post("/api/podcast/answer/stream")
def podcast_answer_stream(req: PodcastAnswerRequest, book: str = "") -> StreamingResponse:
    """插嘴（流式）：边生成边按「轮」吐 NDJSON，前端第一句好了就合成播放，压低首响延迟。

    事件：{"type":"turn","v":"小晓：","t":"…"}，最后 {"type":"done"}；出错 {"type":"error","error":"…"}。
    """
    book = book or _first_book()
    chunks = get_store().search(req.question, book=book or None) if book else []

    def gen():
        if not book:
            yield json.dumps({"type": "error", "error": "还没有导入书"}, ensure_ascii=False) + "\n"
        elif not chunks:
            yield json.dumps({"type": "error", "error": "没检索到相关内容"}, ensure_ascii=False) + "\n"
        else:
            try:
                for voice, text in stream_podcast_answer(req.question, chunks, req.context or ""):
                    yield json.dumps({"type": "turn", "v": voice, "t": text}, ensure_ascii=False) + "\n"
            except Exception as e:  # noqa: BLE001
                yield json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/notes/download")
def download_notes(book: str = ""):
    book = book or _first_book()
    path = _notes_path(book) if book else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="还没有笔记")
    return FileResponse(path, filename=f"{book}.md", media_type="text/markdown")


# ---------- 会话（按书存） ----------
@app.get("/api/session")
def get_session(book: str = "") -> dict:
    book = book or _first_book()
    if not book:
        return {"messages": []}
    path = _session_path(book)
    try:
        if path.exists():
            return {"messages": json.loads(path.read_text(encoding="utf-8"))}
    except Exception:  # noqa: BLE001
        pass
    return {"messages": []}


@app.post("/api/session")
def save_session(req: SessionRequest, book: str = "") -> dict:
    book = book or _first_book()
    if not book:
        return {"ok": False, "error": "还没有导入书"}
    try:
        config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        _session_path(book).write_text(json.dumps(req.messages, ensure_ascii=False), encoding="utf-8")
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def main() -> int:
    import uvicorn

    get_store()  # 预热模型，避免第一次提问卡十几秒
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
