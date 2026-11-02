# mbbank-captcha-solver

Repository cung cấp mô hình CNN **đã được huấn luyện để nhận dạng (giải) ảnh CAPTCHA của `online.mbbank.com.vn`**, kèm HTTP API để sử dụng mô hình. Trọng số có sẵn trong `mbbank/model.pth`, không cần train lại để chạy dự đoán.

> Dự án độc lập, không phải sản phẩm chính thức của MBBank. Chỉ sử dụng với dữ liệu và hệ thống mà bạn có quyền xử lý, tuân thủ điều khoản của dịch vụ. API này chỉ nhận ảnh và trả về ký tự dự đoán, không thực hiện đăng nhập hay giao dịch ngân hàng.

## Mô hình

- Framework: PyTorch, kiến trúc CNN.
- Đầu vào được chuyển sang RGB và resize về **160 × 50 pixel**.
- Đầu ra: **6 ký tự**, có phân biệt chữ hoa và chữ thường.
- API chạy suy luận trên CPU và nạp trọng số một lần khi khởi động.
- Không có số liệu benchmark độc lập trong repo; không bảo đảm mọi ảnh đều được nhận dạng chính xác hoặc mô hình vẫn phù hợp nếu định dạng CAPTCHA thay đổi.
- Dataset huấn luyện không được đính kèm. `mbbank/MAIN.py` chứa mã mô hình và mã huấn luyện; không cần chạy file này để dùng API.

## Nguồn dữ liệu và kết quả huấn luyện

Theo thông tin do chủ repository cung cấp:

- **Dataset huấn luyện:** [MBBank Captcha Images — cookiegmvn trên Kaggle](https://www.kaggle.com/datasets/cookiegmvn/mbbank-captcha-images).
- **Ngày huấn luyện:** 02/11/2026 (ngày 2 tháng 11 năm 2026).
- **Độ chính xác được báo cáo của mô hình:** **99.95%**.

Repo hiện chưa đính kèm báo cáo đánh giá, cách chia tập train/validation/test hoặc định nghĩa metric (đúng toàn bộ CAPTCHA hay đúng từng ký tự). Vì vậy, con số trên là kết quả được chủ repository cung cấp, chưa được kiểm chứng độc lập và không bảo đảm độ chính xác tương tự trên dữ liệu mới.

## Chạy bằng Docker

Cần Docker với Linux containers. Từ thư mục gốc của repo, build image CPU `linux/amd64`:

```sh
docker build --platform linux/amd64 -t mbbank-captcha-solver:local .
MBCAPTCHA_IMAGE=mbbank-captcha-solver:local docker compose up -d --wait
```

Trên máy ARM, image này cần hỗ trợ giả lập `amd64`. Compose chỉ publish API tại `127.0.0.1:8000`.

Kiểm tra và dừng dịch vụ:

```sh
curl --fail http://127.0.0.1:8000/health
MBCAPTCHA_IMAGE=mbbank-captcha-solver:local docker compose logs --tail=100 api
MBCAPTCHA_IMAGE=mbbank-captcha-solver:local docker compose down
```

## Chạy bằng Python

Dùng Python **3.11** và môi trường tương thích với các phiên bản đã pin trong `mbbank/requirements.txt`:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r mbbank/requirements.txt
.venv/bin/python -m flask --app mbbank.server run --host 127.0.0.1 --port 8000
```

Flask development server chỉ dành cho phát triển local. Để chạy dịch vụ bằng Gunicorn:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 4 \
  --timeout 60 mbbank.server:app
```

## HTTP API

| Method | Endpoint | Chức năng |
| --- | --- | --- |
| `GET` | `/` | Thông tin dịch vụ và các kiểu đầu vào |
| `GET` | `/health` | Trả về `{"status":"ok"}` sau khi mô hình được nạp |
| `POST` | `/predict` | Nhận ảnh và trả về `{"prediction":"..."}` |

### Upload ảnh

Repo có ảnh mẫu `temp.png`. Từ thư mục gốc:

```sh
curl --fail-with-body http://127.0.0.1:8000/predict \
  -F 'image=@temp.png'
```

Kết quả là JSON chứa chuỗi dự đoán 6 ký tự. Thay `temp.png` bằng ảnh bạn có quyền sử dụng.

### Gửi JSON Base64 bằng Python

```python
import base64
from pathlib import Path

import requests

image = Path("temp.png").read_bytes()
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"image_base64": base64.b64encode(image).decode("ascii")},
    timeout=30,
)
response.raise_for_status()
print(response.json()["prediction"])
```

`image_base64` chấp nhận cả Base64 thuần và data URI dạng `data:image/png;base64,...`.

### Giới hạn và bảo mật

- Request body tối đa **2 MiB**, tính cả multipart/JSON overhead.
- Ảnh tối đa **1.000.000 pixel** trước khi resize.
- Đầu vào không hợp lệ trả HTTP `400`; request quá lớn trả HTTP `413`.
- API hiện **không có xác thực**. Không mở trực tiếp ra Internet; nếu cần chia sẻ, hãy bổ sung xác thực, TLS và giới hạn truy cập ở lớp triển khai của bạn.
- Không commit IP máy chủ, thông tin đăng nhập hoặc khoá bí mật. Tài liệu và cấu hình triển khai riêng được loại trừ bằng `.gitignore`.

## Kiểm thử

Từ thư mục gốc, sau khi cài dependencies:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Với API đang chạy local:

```sh
.venv/bin/python tests/smoke_api.py http://127.0.0.1:8000 \
  --image temp.png --check-size-limit
```

Các bài test kiểm tra API, tính nhất quán của suy luận, đầu vào lỗi và request đồng thời; không thay thế benchmark độ chính xác trên tập CAPTCHA có nhãn.