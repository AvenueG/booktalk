"""booktalk — 个人用「可打断的书本语音问答」。

当前实现 M2 核心：书导入 + 带引用的文字问答（大脑 = DeepSeek）。
语音层（M1）单独在 server/ 里，跑在 Windows 4080S。
"""

__version__ = "0.1.0"

import os

# 必须最先设置：huggingface_hub 在 import 时就会读取并锁定 HF_ENDPOINT，
# 所以要在任何会 import 它的库（docling / sentence-transformers）之前设好。
# 部分地区连不上 huggingface.co，默认走镜像；可用环境变量覆盖。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
