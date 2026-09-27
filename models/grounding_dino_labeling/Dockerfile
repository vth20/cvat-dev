FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=false \
    GRADIO_INBROWSER=false

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và xử lý ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng vào container
COPY app_gradio.py inference.py metrics_eval.py ./

# Tạo thư mục xuất kết quả & dữ liệu
RUN mkdir -p /app/gradio_exports /app/data

# Mở cổng Gradio WebUI
EXPOSE 7860

# Kiểm tra trạng thái container (Healthcheck)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Lệnh khởi chạy mặc định (Giao diện Gradio WebUI)
CMD ["python", "app_gradio.py"]
