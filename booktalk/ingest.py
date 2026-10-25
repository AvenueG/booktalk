"""导入一本书：Docling 解析 -> 按章节分块 -> 写向量库。

用法：
    python -m booktalk.ingest [--reset] path/to/book.pdf
    --reset 先清空旧索引，只保留这本新书（换书用）。
    支持 pdf / epub / md / txt（docling 还能处理 docx/html 等）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter

from . import config
from .store import BookStore, Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_by_headings(markdown: str, source: str) -> list[Chunk]:
    """按标题切块；过长的块按 CHUNK_MAX_CHARS 滑窗再切（保留同章节出处）。"""
    current_heading = "（全书开头）"
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        if len(text) <= config.CHUNK_MAX_CHARS:
            chunks.append(Chunk(text=text, source=source, heading=current_heading))
        else:
            step = config.CHUNK_MAX_CHARS - config.CHUNK_OVERLAP_CHARS
            for i in range(0, len(text), step):
                piece = text[i : i + config.CHUNK_MAX_CHARS].strip()
                if piece:
                    chunks.append(Chunk(text=piece, source=source, heading=current_heading))

    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if m and len(line) <= 120:
            flush()
            current_heading = m.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    return chunks


_TEXT_EXTS = {".md", ".markdown", ".txt", ".text"}


def _convert_to_markdown(path: Path) -> str:
    # 纯文本/标记文件直接读，不必走 Docling（省得下载解析模型）
    if path.suffix.lower() in _TEXT_EXTS:
        return path.read_text(encoding="utf-8")
    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def main() -> int:
    reset = "--reset" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--reset"]
    if not args:
        print("用法: python -m booktalk.ingest [--reset] <书文件.pdf|.epub|.md|.txt>")
        return 1
    path = Path(args[0])
    if not path.exists():
        print(f"文件不存在: {path}")
        return 1

    config.BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"解析 {path.name} …（首次会下载 Docling 解析模型）")
    markdown = _convert_to_markdown(path)
    chunks = _split_by_headings(markdown, path.name)
    print(f"切出 {len(chunks)} 块，写入向量库 …（首次会下载 bge-m3 / 重排模型）")
    store = BookStore()
    if reset:
        store.clear()
        print("已清空旧索引，只保留这本新书。")
    n = store.add_chunks(chunks)
    print(f"完成，已写入 {n} 块。现在运行 python -m booktalk.chat 提问。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
