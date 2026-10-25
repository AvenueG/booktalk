# 代码审查（v0.1.0 上线前）

按优先级排序，从「会影响正确性」到「锦上添花」。每条都给了具体位置和改法。

## P0 · 正确性 bug（已修复）

1. **`web.py` `podcast_route` 的 `cache` 变量未初始化**
   - 现象：一本书**首次**生成播客（`data/podcast/{book}.json` 还不存在）时，`cache` 从未被赋值，走到 `cache[key] = …` 直接 `NameError`，被外层 `except` 吃掉，前端只看到 `{"ok": false, "error": "name 'cache' is not defined"}`。
   - 根因：`cache = {}` 只写在 `except` 分支里；`cache_path.exists()` 为 False 时既不进 `if` 也不抛异常，`cache` 悬空。
   - 修法（已改）：`cache_path = …` 之后先 `cache = {}`。
   - 教训：**缓存读/写要先把默认值放在最前**，别依赖某个分支去赋值。

## P1 · 检索（性能 + 正确性）

2. **`store.py` `search` 的分书过滤放在 Python 端**
   ```python
   limit = config.TOP_K_RETRIEVE if not book else max(config.TOP_K_RETRIEVE * 8, 200)
   hits = self.client.query_points(..., limit=limit, ...).points
   if book:
       hits = [h for h in hits if (h.payload or {}).get("book") == book]
   ```
   多书时先取 200 条再在内存里筛。风险：目标书的相关块若排不进前 200，会被**漏掉**（检索不全）；且每次多取、多传。应改成 Qdrant 服务端过滤：
   ```python
   from qdrant_client import models
   flt = models.Filter(must=[models.FieldCondition(key="book", match=models.MatchValue(value=book)]) if book else [])
   hits = self.client.query_points(collection_name=config.COLLECTION, query=q.tolist(),
                                   query_filter=flt, limit=config.TOP_K_RETRIEVE, with_payload=True).points
   ```
   这样 `limit` 永远只需 `TOP_K_RETRIEVE`，且不会漏。

3. **`store.py` `add_chunks` 的进度条会污染日志**
   `embedder.encode(..., show_progress_bar=True)` 在网页端导入时会往 stdout 刷进度条。改成 `show_progress_bar=False`（或只在 CLI 路径传 True）。

## P1 · 重复代码（DRY）

4. **「小晓：/小宇：」的轮次切分写了三遍**
   `brain._split_turns`、`brain.stream_podcast_answer`、`web._parse_turns` 各自实现一遍正则 + 状态机。抽成一个共享函数：
   ```python
   def parse_host_turns(text: str) -> list[tuple[str, str]]:
       """把「小晓：…/小宇：…」对白切成 [(主播标签, 文本), …]；没前缀的行归上一人。"""
   ```
   `web` 层再把「主播标签 → edge-tts 音色」映射一下即可。好处：以后想改主播前缀/加主播，只改一处。

## P2 · 结构（单文件巨石，刻意的取舍）

5. **`static/index.html`（~2400 行）与 `web.py`（~840 行）是单文件**
   这是「单文件可部署、方便复制改」的刻意取舍，**保留**。但建议立个阈值：`index.html` 超过 ~4000 行或 `web.py` 超过 ~1500 行时，按 `store / brain / routes / 前端组件` 拆。现在不拆。

## P2 · 错误处理 / 日志

6. **大量静默吞错，没有 traceback**
   到处是 `except Exception: pass` / `except Exception: return {"ok": False, "error": str(e)}`。开源后别人报 bug，你手里只有一句前端提示、没有服务端现场。建议：
   - 引入标准 `logging`（uvicorn 自带 handler，`logging.getLogger("booktalk")` 即可）。
   - 这些裸 `except` 里加 `logging.exception("...")`。
   - `web.py` 里的 `print(...)`（`_migrate_legacy`、`main`）换成 `logging.info`。

7. **启动时不校验 API key**
   `web.main()` 直接 `get_store()` 预热模型，但没检查 key；缺 key 时首次提问才报「出错了：…」。建议启动时：
   ```python
   if not config.get_deepseek_key():
       logging.warning("未配置 API key：可在网页右上角 ⚙ 设置里填，或 cp .env.example .env")
   ```

## P2 · 安全 / 健壮性（单机单用户，低危但值得挡）

8. **`/api/import` 用 `file.filename` 拼路径，可被路径穿越**
   `tmp = config.BOOKS_DIR / ("_upload_" + filename)`、`book_name = Path(filename).stem`。恶意文件名如 `../../x` 会写出 `data/books/` 之外（本地单用户风险低，但值得挡）。修法：
   ```python
   filename = Path(file.filename or "book").name   # 只留最后一段，去路径
   ```

9. **127.0.0.1 + 无鉴权是既定取舍**
   这个取舍没问题（个人单用户），但**必须在 README 里显式写清**：只在本机跑，不要 `--host 0.0.0.0` 暴露公网，否则任何人都能读你的书、用你的 key。

## P3 · 测试

10. **只有 `scripts/smoke.py`，无断言、无 CI**
    开源前至少补一条**不依赖真实 API key** 的冒烟：导入（用小 md/txt 即可）→ 分块 → 检索 → 笔记落盘全链路，LLM 调用用 mock 或直接跳过。并在 README 写「如何自测」。可选：GitHub Actions 跑 `python -m py_compile` + 冒烟。

## P3 · 细节

11. **`config.set_api_key` 直接重写 `data/config.json`，并发写会竞争**
    单用户概率极低；若要更稳，用「临时文件 + `os.replace`」原子写。

12. **`store._scroll_all` 全量扫描**
    `indexed_books` / `clear` / `tag_legacy_points` 都用它，O(N)。单用户可接受；书多了可换 `scroll` 的 payload 过滤器或 Qdrant 原生 filter。

---

## 做得好的地方（保持）

- **Prompt 分层清晰**：`_BASE_RULES`（不可破的引用底线）+ 人设 `tone/example`（只改语气）+ 播客主播人设（`_PODCAST_HOSTS`）分离得很干净，这是最容易「越改越乱」的部分，现在反而最稳。
- **懒加载压启动时间**：`docling` / `jieba` / `edge_tts` 都延迟 import，网页首次打开快很多。
- **流式 + 兜底**：`stream_podcast_answer` 的 `yielded == 0` 退一次性调用兜底，直接解决了「模型不按前缀出牌就静音」的坑，实用。
- **缓存思路**：中心句 / 播客脚本按「书 + 参数」缓存，重放不花 API 钱，正对 $3.5 预算。
