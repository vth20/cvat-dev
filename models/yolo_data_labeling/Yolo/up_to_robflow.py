import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Tự động nạp cấu hình từ file .env (tìm ở thư mục hiện tại hoặc thư mục gốc dự án)
load_dotenv()

# 1. Đọc API Key từ biến môi trường
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("Chưa cấu hình ROBOFLOW_API_KEY trong file .env hoặc biến môi trường!")

rf = Roboflow(api_key=api_key)

# 2. Đọc tên Workspace và Project
workspace_name = os.getenv("ROBOFLOW_WORKSPACE", "chung-do-quoc")
project_name = os.getenv("ROBOFLOW_PROJECT", "window-demo")
project = rf.workspace(workspace_name).project(project_name)

# 3. Chọn Version dataset bạn muốn gắn model vào
version_num = int(os.getenv("ROBOFLOW_MODEL_VERSION", 1))
version = project.version(version_num)

# 4. Upload model lên Roboflow
# Lưu ý quan trọng: model_path trỏ tới thư mục CHỨA thư mục weights (ví dụ: ./thu_muc_cua_ban/)
model_path = os.getenv("ROBOFLOW_MODEL_PATH", "E:/upload_roboflow/")
version.deploy(model_type="yolov11", model_path=model_path)