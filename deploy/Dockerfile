# 推理服务镜像（CPU only）。
#
# 为什么不带 torch：服务端只做前向，torch 的 2~3 GB 依赖完全没有收益。
# 实测同硬件 ONNX Runtime 前向比 PyTorch CPU 快 1.66×（见 eval_reports/deploy_speed.md），
# 所以这里只装 onnxruntime。
#
# 构建（在仓库根目录执行，context 必须是根，因为要拷 tools/ 和 src/）：
#   docker build -f deploy/Dockerfile -t cabbage-detect:0.1.0 .
# 运行：
#   docker run --rm -p 8000:8000 cabbage-detect:0.1.0

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/deploy/models/base_v2.onnx \
    IMGSZ=640 \
    MAX_WORKERS=4 \
    INTRA_THREADS=3

WORKDIR /app

# opencv-headless 仍需要 glib；libgl1 不用装（headless 版不带 GUI 代码）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements-serve.txt /app/deploy/requirements-serve.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements-serve.txt

# 只拷服务真正需要的东西，不把 dataset/runs/论文/图表带进镜像
COPY src/ /app/src/
COPY tools/__init__.py tools/onnx_common.py /app/tools/
COPY deploy/models/base_v2.onnx /app/deploy/models/base_v2.onnx

EXPOSE 8000

# 容器内自检：模型没加载成功时容器应被标记为 unhealthy，而不是等业务请求报错
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
