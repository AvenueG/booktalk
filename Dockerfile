FROM python:3.11-slim

# docling / OpenCV 解析 PDF 可能用到的系统库（图片 / 字体渲染）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 模型缓存与数据都走挂载卷：重建容器不重新下载模型、不丢书
ENV HF_HOME=/root/.cache/huggingface \
    BOOKTALK_HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["python", "-m", "booktalk.web"]
