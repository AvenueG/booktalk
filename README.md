# booktalk

一个**个人单用户**的本地「读书搭子」：把 PDF / EPUB / 电子书丢进来，用你自己的 DeepSeek API Key，在浏览器里**读、聊、追问、做笔记、听播客**——像 NotebookLM，但数据都在你自己机器上。

- **带引用的问答**：答案标注出处 `[n]`，点击跳回书中对应章节；书外常识标「（书外补充）」，不糊弄。
- **可打断的播客**：双主播「小晓 / 小宇」对谈，播到一半点 🎤 插嘴，主播答完自然接回。
- **读 + 记一体**：高亮、书签（进度条上有刻度）、批注、所见即所得笔记，还能一键按章节整理。

> ⚠️ **只在自己机器上跑。** 服务默认绑定 `127.0.0.1`，**不要**用 `--host 0.0.0.0` 暴露到公网——没有登录鉴权，暴露等于把你的书和 API Key 公开。

## 功能

| 类别 | 内容 |
| --- | --- |
| 书 | PDF / EPUB / MD / TXT / DOCX 导入（Docling 解析）；多本书切换 |
| 问答 | DeepSeek `deepseek-chat` + 本地 RAG（bge-m3 嵌入 + bge-reranker 重排），流式输出 |
| 追问 | 线程式追问，锚定在父回答的出处上，不跑偏 |
| 人设 / 深度 | 读书搭子 / 萝莉 / 严谨学者 / 浪漫；标准 / 简短 / 深入 |
| 阅读辅助 | 跟着读（聊聊 / 考考 / 复盘）、中心句、日间 / 暗夜 / 手电筒 |
| 标注 | 高亮、书签（进度条刻度）、批注、笔记（自动保存 + 按章节整理 + 导出） |
| 播客 | 深度对谈 / 简报 / 批判点评 / 辩论，双音色合成，可保存；插嘴打断 |
| 语音 | 浏览器语音输入（push-to-talk）、edge-tts 朗读 |
| 设置 | 网页右上角 ⚙ 里填自己的 API Key，选人设 / 深度 / 显示模式 |

## 快速开始

```bash
cd booktalk
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 启动（网页版）
python -m booktalk.web
# 浏览器打开 http://127.0.0.1:8000
```

首次启动 / 首次导入会下载模型（bge-m3 嵌入 ~2GB、bge-reranker ~2.3GB、Docling 解析模型），**耐心等一次**；之后都在本地跑。

### 填 API Key（两种方式任选）

1. **网页里填（推荐）**：打开页面 → 右上角 **⚙ 设置** → 粘贴你的 DeepSeek API Key → 保存。Key 只存在本机 `data/config.json`（已 gitignore）。
2. **`.env` 里填**：`cp .env.example .env`，编辑填 `DEEPSEEK_API_KEY=sk-...`。

> 每个用的人填**自己的** Key。没有 Key 去 https://platform.deepseek.com 申请。

### 命令行（可选）

```bash
python -m booktalk.ingest "你的书.pdf"     # 导入一本书（网页端也可直接上传）
python -m booktalk.chat                   # 纯文字问答 CLI
```

## 成本

大脑用 `deepseek-chat(V3)`（约 $0.27/M 输入、$1.10/M 输出），一次问答约 $0.003–0.006，**$3.5 够用数月**。播客脚本按「书 + 参数」缓存，重放不重复花钱；中心句 / 提纲用本地 jieba 抽取，不耗 API。

⚠️ 别用 `deepseek-reasoner`/R1——推理延迟会毁掉语音和插嘴的体验。

## 目录结构

```
booktalk/
  booktalk/
    ingest.py   # 导入：Docling 解析 -> 按章节分块 -> 写向量库
    store.py    # Qdrant 本地 + bge-m3 嵌入 + bge-reranker 重排
    brain.py    # DeepSeek 回答 / 人设 / 笔记整理 / 播客脚本
    web.py      # FastAPI 后端
    chat.py     # 文字问答 CLI
    config.py   # 配置 / 路径 / 运行时 API key
    static/     # 单页前端
  personas/     # 人设（可扩展，JSON）
  scripts/      # smoke.py 冒烟
  docs/         # REVIEW.md 代码审查建议
  data/         # 书 / 向量库 / 笔记 / 会话 / 播客（自动生成，已 gitignore）
```

## 技术栈

FastAPI · Qdrant（本地 path 模式）· bge-m3 / bge-reranker-v2-m3 · Docling · DeepSeek（OpenAI 兼容接口）· edge-tts · 浏览器 Web Speech API · Vditor。

## 协议

[PolyForm Noncommercial License 1.0.0](LICENSE) —— 个人 / 学习 / 研究 / 非商业使用与修改可以，**禁止商用**。

## 已知取舍 / 下一步

- 高亮 / 书签 / 批注存在浏览器 localStorage（单设备），笔记 / 会话存在服务端文件。
- 语音插嘴目前用浏览器 Web Speech（延迟与音色有限）；更低的实时延迟、音色区分、按句续播是后续方向。
- 详见 [`docs/REVIEW.md`](docs/REVIEW.md) 的代码审查建议与 [`CHANGELOG.md`](CHANGELOG.md)。
