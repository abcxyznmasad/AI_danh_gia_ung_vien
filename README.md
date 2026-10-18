# Đánh giá Quy trình Sàng lọc Ứng viên IT: Tính Nhất quán, Công bằng và Dự đoán Mức Phù hợp bằng Python

Đồ án môn **Python Ứng dụng** thực hiện kiểm toán dữ liệu tuyển dụng IT (`12.000` hồ sơ, `20` vị trí công việc, `28` thuộc tính) thông qua phân tích thống kê chuyên sâu và mô hình học máy có khả năng diễn giải.

## 1. Cấu trúc thư mục

```text
project/
├── app.py                        # Ứng dụng giao diện Streamlit (7 tab)
├── data/
│   └── candidates_full.csv       # Tập dữ liệu 12.000 ứng viên IT
├── modules/
│   ├── __init__.py
│   ├── data_processing.py        # Đọc, làm sạch, kiểm tra chất lượng và chuẩn bị đặc trưng
│   ├── analysis.py               # Giải quyết 3 câu hỏi nghiên cứu & kiểm định thống kê
│   ├── modeling.py               # Baseline, Logistic Regression, Linear Regression, Fairness, SHAP
│   └── visualizations.py         # Toàn bộ hàm vẽ biểu đồ trả về matplotlib.figure.Figure
├── tests/
│   ├── test_data_processing.py   # Unit test cho làm sạch dữ liệu & chống rò rỉ
│   └── test_analysis.py          # Unit test cho các hàm thống kê & kiểm định
├── requirements.txt              # Danh sách thư viện với phiên bản cố định
└── README.md                     # Hướng dẫn cài đặt và chạy dự án

## Hướng dẫn cài đặt và chạy dự án

### Yêu cầu

- Windows 10/11.
- Python đã được cài đặt và có thể chạy bằng lệnh `python`.
- Kết nối Internet khi cài đặt thư viện.

### Cài đặt tự động

1. Tải hoặc sao chép mã nguồn dự án về máy.
2. Mở thư mục gốc chứa `app.py` và `requirements.txt`.
3. Chạy file `setup.bat`.
4. Chờ quá trình cài thư viện và kiểm thử hoàn tất.

File `setup.bat` sẽ tự tạo môi trường ảo `venv` nếu cần, cài đặt các thư viện được khai báo trong `requirements.txt` và chạy bộ kiểm thử tự động.

### Khởi chạy ứng dụng

Mở PowerShell tại thư mục gốc dự án và chạy:

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

Sau đó mở địa chỉ Local URL được Streamlit hiển thị trên terminal.

### Chạy kiểm thử

```powershell
venv\Scripts\python.exe -m pytest -v
```
