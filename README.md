# booktalk

个人用「可打断的书本语音问答」——复刻 NotebookLM 的 Interactive 音频概览体验。

**现状**：本仓库当前实现 **M2 核心**——把一本书导入、建向量索引、用 DeepSeek 做**带引用出处**的文字问答。语音层（M1）下一步在 Windows 4080S 上接。

## 路线图

- [ ] **M0 · 免费体验定标准（0.5 天，不写代码）**——见下方清单
- [x] **M2 · 书的记忆（文字问答，带引用）**——本仓库已实现
- [ ] **M1 · 实时语音地基**——LiveKit + FunASR + DeepSeek + Qwen3-TTS（跑在 Windows 4080S）
- [ ] **M3 · 合并**——实时语音 + RAG + 引用 + 文字兜底 UI
- [ ] **M4 · 移动端**——PWA + Tailscale 远程
- [ ] **M5 · 可选双人播客**

## M0 清单（先做，不花钱）

用免费的 AI 语音 App（ChatGPT 语音模式、NotebookLM 若能访问等）体验「可打断」，记录你要的标准：

- [ ] 首字延迟多少秒内你觉得「跟得上」？
- [ ] 打断后它要多久停、会不会「抢话」？
- [ ] 音色要真人感还是清晰即可？
- [ ] 中英文切换顺不顺？

## 快速开始（M2 文字问答）

```bash
cd booktalk
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) 配 DeepSeek key
cp .env.example .env    # 然后编辑 .env 填入 DEEPSEEK_API_KEY

# 2) 把一本书放进 data/books/，导入
python -m booktalk.ingest "data/books/你的书.pdf"    # 支持 pdf/epub/md/txt

# 3) 提问
python -m booktalk.chat
```

首次运行会下载模型（bge-m3 嵌入 ~2GB、bge-reranker ~2.3GB、Docling 解析模型），耐心等一次。

## 目录

```
booktalk/
  booktalk/
    ingest.py   # 导入：Docling 解析 -> 按章节分块 -> 写向量库
    store.py    # Qdrant 本地 + bge-m3 嵌入 + bge-reranker 重排
    brain.py    # DeepSeek（deepseek-chat）带引用回答
    chat.py     # 文字问答 CLI
    config.py   # 配置/路径
  data/books/   # 放书
  data/index/   # 向量库（自动生成）
```

## 成本

大脑用 `deepseek-chat(V3)`（约 $0.27/M 输入、$1.10/M 输出），一次问答 ≈ $0.003–0.006，$3.5 够用数月。**别用 R1/deepseek-reasoner**——推理延迟会毁掉后续语音体验。

## 下一步（M1）

在 Windows 4080S 上搭实时语音链路（LiveKit Agents + FunASR ASR + 本地 Qwen3-TTS），大脑复用这里的 `brain.py`。参见计划文件 `~/.claude/plans/crispy-twirling-castle.md`。
