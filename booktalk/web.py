"""本地网页版：FastAPI 后端 + 单页前端。

用法：
    python -m booktalk.web
    然后浏览器打开 http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import config
from .brain import answer, list_personas, load_persona, stream_answer, suggest
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


class SessionRequest(BaseModel):
    messages: list[dict] = []


# ---------- 页面 ----------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    path = Path(__file__).parent / "static" / "index.html"
    return path.read_text(encoding="utf-8")


@app.get("/api/personas")
def personas() -> list[dict]:
    return list_personas()


# ---------- 书 ----------
@app.get("/api/books")
def books() -> list[dict]:
    return [{"name": b} for b in _books()]


@app.get("/api/book/{name}")
def book(name: str) -> dict:
    return {"name": name, "markdown": _book_markdown(name)}


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
        n = import_book(tmp, book=book_name, store=store)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"ok": True, "name": book_name, "chunks": n}


# ---------- 问答 ----------
@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    store = get_store()
    chunks = store.search(req.question, book=req.book or None)
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
    chunks = store.search(req.question, book=req.book or None)
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
    chunks = get_store().search(query, book=req.book or None) if query else []
    if not chunks:
        return {"text": ""}
    persona = load_persona(req.persona)
    instruction = (
        "你正在以播客主播的方式陪用户聊这本书。顺着刚才的话自然往下讲："
        "可以展开一个细节、补一个生动的例子，或自然过渡到相邻话题。"
        "像播客一样自说自话、有语气、有画面感；不要向用户提问、不要等用户回应。"
        "控制在 2-4 句，用与用户相同的语言。"
    )
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
