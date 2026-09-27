# 🏛️ Zero-Shot Facade Window Detection (Grounding DINO)

Hệ thống nhận diện cửa sổ mặt đứng kiến trúc tự động bằng mô hình **Grounding DINO** (`IDEA-Research/grounding-dino-base`).
- Tích hợp giao diện **WebUI Gradio** trực quan, nhận diện hàng loạt ảnh cùng lúc.
- Tự động xuất và đóng gói bộ nhãn theo chuẩn **YOLO (.txt)** và **COCO JSON (`annotations.coco.json`)** dưới dạng file `.zip`.
- Tự động tải trọng số mô hình từ **Hugging Face Hub** trong lần chạy đầu tiên, tối ưu hóa tăng tốc GPU CUDA.

---

## 📂 Cấu trúc thư mục

```text
grounding_dino_labeling/
├── app_gradio.py           # Ứng dụng giao diện WebUI Gradio (Nhận diện & đóng gói nhãn)
├── inference.py            # Script chạy nhận diện tự động qua dòng lệnh (CLI)
├── metrics_eval.py         # Bộ công cụ đánh giá độ chính xác (Precision, Recall, mAP)
├── requirements.txt        # Danh sách các thư viện Python phụ thuộc
├── Dockerfile              # Cấu hình Dockerfile đóng gói ứng dụng
├── .dockerignore           # Danh sách các file bỏ qua khi đóng gói Docker
├── .gitignore              # Danh sách các file bỏ qua khi commit Git
└── README.md               # Tài liệu hướng dẫn chi tiết
```

> 💡 **Cơ chế tải Model tự động**: Mô hình Grounding DINO (`IDEA-Research/grounding-dino-base`) sẽ tự động được tải từ Hugging Face Hub về bộ nhớ đệm cache trong lần chạy đầu tiên.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

Bạn có thể lựa chọn **1 trong 2 cách** dưới đây để chạy hệ thống:

---

### 🔹 Cách 1: Chạy bằng Docker Image (Khuyến nghị — Nhanh nhất)

Image đã được đóng gói sẵn toàn bộ môi trường và tải lên Docker Hub:  
👉 **`nguyenhoangkhang040904/window-detection:latest`**

#### 1. Chạy với GPU NVIDIA (Khuyến nghị để đạt tốc độ cao nhất)
> Yêu cầu máy chủ có GPU NVIDIA và đã cài đặt [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker run -d \
  --name window_detection_app \
  --gpus all \
  --restart unless-stopped \
  -p 7860:7860 \
  -v $(pwd)/gradio_exports:/app/gradio_exports \
  nguyenhoangkhang040904/window-detection:latest
```

#### 2. Chạy trên CPU (Nếu không có card GPU)
```bash
docker run -d \
  --name window_detection_app \
  --restart unless-stopped \
  -p 7860:7860 \
  -v $(pwd)/gradio_exports:/app/gradio_exports \
  nguyenhoangkhang040904/window-detection:latest
```

> 💡 **Lưu ý cú pháp thư mục mount (`-v`):**
> - **Linux / macOS / WSL**: `$(pwd)/gradio_exports:/app/gradio_exports`
> - **Windows PowerShell**: `${PWD}/gradio_exports:/app/gradio_exports`
> - **Windows Command Prompt (CMD)**: `%cd%/gradio_exports:/app/gradio_exports`

Sau khi khởi chạy, mở trình duyệt truy cập: **`http://localhost:7860`** (hoặc `http://127.0.0.1:7860`).

---

### 🔹 Cách 2: Cài đặt và Chạy trên Môi trường Python Cục bộ

#### Bước 1: Tạo và kích hoạt môi trường ảo

- **Sử dụng Conda (Khuyến nghị Python 3.10)**:
  ```bash
  # Tạo môi trường tên cv_torch_env
  conda create -n cv_torch_env python=3.10 -y

  # Kích hoạt môi trường
  conda activate cv_torch_env
  ```

- **Hoặc sử dụng `venv`**:
  ```bash
  # Tạo môi trường
  python -m venv cv_torch_env

  # Kích hoạt trên Linux / WSL / macOS:
  source cv_torch_env/bin/activate

  # Kích hoạt trên Windows (CMD / PowerShell):
  cv_torch_env\Scripts\activate
  ```

#### Bước 2: Cài đặt PyTorch và các thư viện phụ thuộc

- **Cài đặt PyTorch phù hợp với phần cứng**:
  - *GPU NVIDIA (CUDA 12.1)*:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```
  - *GPU NVIDIA (CUDA 11.8)*:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
  - *Chạy trên CPU*:
    ```bash
    pip install torch torchvision
    ```

- **Cài đặt các thư viện từ `requirements.txt`**:
  ```bash
  pip install -r requirements.txt
  ```

#### Bước 3: Khởi chạy ứng dụng

- **Khởi động Giao diện Web (WebUI Gradio)**:
  ```bash
  python app_gradio.py
  ```
  Truy cập vào địa chỉ hiển thị trên terminal (mặc định: `http://localhost:7860`).

- **Hoặc chạy nhận diện hàng loạt qua dòng lệnh (CLI - `inference.py`)**:
  - *Nhận diện 1 ảnh lẻ*:
    ```bash
    python inference.py --input "duong/dan/toi/anh.jpg" --output "./ket_qua"
    ```
  - *Nhận diện cả thư mục ảnh*:
    ```bash
    python inference.py --input "duong/dan/toi/thu_muc_anh" --output "./ket_qua"
    ```

---

## 🖥️ Hướng dẫn Sử dụng Giao diện WebUI

1. **Tải ảnh lên**: Kéo thả hoặc chọn một/nhiều ảnh mặt đứng kiến trúc tại ô **"📁 Tải lên ảnh"**.
2. **Bắt đầu nhận diện**: Nhấn nút **"🚀 Bắt Đầu Nhận Diện & Đóng Gói Nhãn"**.
3. **Xem kết quả trực quan**: Quan sát các bounding box cửa sổ được phát hiện tại khung **"🖼️ Kết quả Dự Đoán Trực Quan"**.
4. **Tải về bộ dữ liệu gán nhãn**:
   - 📥 **Tải về Bộ Nhãn YOLO (.zip)**: Chứa thư mục `images/`, `labels/` và file `classes.txt`.
   - 📥 **Tải về Bộ Nhãn COCO (.zip)**: Chứa file `annotations.coco.json` chuẩn định dạng COCO Object Detection.
