# Credit Default Risk Prediction System

Hệ thống dự đoán nguy cơ vỡ nợ tín dụng cá nhân sử dụng học máy và triển khai trên Docker Swarm.

## Tổng quan hệ thống

Hệ thống này sử dụng các kỹ thuật học máy để dự đoán khả năng khách hàng sẽ gặp khó khăn tài chính và vỡ nợ tín dụng trong 2 năm tới. Dự án bao gồm:

1. **Xử lý dữ liệu**: Làm sạch, xử lý dữ liệu khách hàng (nhân khẩu học, lịch sử tín dụng, thu nhập, nợ)
2. **Mô hình học máy**: Huấn luyện và tối ưu hóa mô hình phân loại rủi ro (Logistic Regression, Random Forest, XGBoost, LGBM)
3. **Giải thích mô hình**: Áp dụng SHAP/LIME để giải thích mô hình
4. **API và giao diện**: REST API và giao diện Streamlit
5. **Triển khai**: Đóng gói với Docker và triển khai trên Docker Swarm

## Cấu trúc dự án

```
UEH_HPC/
├── data/                  # Dữ liệu
│   ├── raw/               # Dữ liệu gốc
│   └── processed/         # Dữ liệu đã xử lý
├── notebooks/             # Jupyter notebooks cho phân tích
├── src/                   # Mã nguồn chính
│   ├── data/              # Xử lý dữ liệu
│   ├── features/          # Kỹ thuật đặc trưng
│   ├── models/            # Mô hình học máy
│   └── visualization/     # Trực quan hóa
├── api/                   # API FastAPI
├── app/                   # Ứng dụng Streamlit
├── models/                # Mô hình đã huấn luyện
├── docker/                # Cấu hình Docker
│   ├── Dockerfile.api     # Dockerfile cho API
│   ├── Dockerfile.app     # Dockerfile cho ứng dụng Streamlit
│   └── Dockerfile.train   # Dockerfile cho huấn luyện mô hình
├── docker-compose.yml     # Cấu hình docker-compose cho phát triển
├── docker-stack.yml       # Cấu hình Docker Swarm
├── deploy.sh              # Script triển khai
└── requirements.txt       # Phụ thuộc Python
```

## Dữ liệu

Dự án sử dụng bộ dữ liệu "Give Me Some Credit" từ Kaggle, chứa thông tin về:

- Thông tin cá nhân (tuổi, số người phụ thuộc)
- Lịch sử tín dụng (số lần trễ hạn, số tài khoản mở)
- Thông tin tài chính (tỷ lệ nợ, thu nhập hàng tháng)
- Nhãn: Khách hàng có gặp khó khăn tài chính trong 2 năm tiếp theo hay không

## Các mô hình đã triển khai

Dự án này triển khai và so sánh 4 mô hình học máy:

1. **Logistic Regression**: Mô hình cơ bản, dễ giải thích
2. **Random Forest**: Mô hình ensemble, xử lý tốt dữ liệu phi tuyến tính
3. **XGBoost**: Thuật toán gradient boosting hiệu suất cao
4. **LightGBM**: Thuật toán gradient boosting hiệu quả và nhẹ

## Cài đặt và triển khai

### Yêu cầu

- Docker và Docker Compose
- Docker Swarm (cho triển khai production)
- Python 3.9 hoặc cao hơn (cho phát triển)

### Phát triển cục bộ

1. Clone repository:
```bash
git clone https://github.com/your-username/UEH_HPC.git
cd UEH_HPC
```

2. Tải dữ liệu:
```bash
mkdir -p data/raw
# Tải dữ liệu từ Kaggle và đặt vào thư mục data/raw
```

3. Chạy ứng dụng với Docker Compose:
```bash
docker-compose up -d
```

4. Huấn luyện mô hình:
```bash
docker-compose run --rm train
```

5. Truy cập giao diện người dùng:
   - UI: http://localhost:8501
   - API docs: http://localhost:8000/docs

### Triển khai trên Docker Swarm

1. Cài đặt Docker Swarm:
```bash
docker swarm init
```

2. Triển khai hệ thống:
```bash
chmod +x deploy.sh
./deploy.sh
```

3. Kiểm tra trạng thái:
```bash
docker stack services credit-risk
```

4. Truy cập giao diện người dùng:
   - UI: http://localhost:8501
   - API: http://localhost:8000
   - Visualizer: http://localhost:8080

## API Endpoints

- **GET /**: Thông tin API
- **GET /models**: Danh sách mô hình có sẵn
- **POST /predict/{model_name}**: Dự đoán rủi ro vỡ nợ
- **POST /compare_models**: So sánh kết quả từ tất cả các mô hình
- **POST /explain/{model_name}**: Giải thích dự đoán

## Giao diện người dùng

Giao diện người dùng Streamlit bao gồm:

1. **Trang dự đoán**: Nhập thông tin khách hàng và nhận dự đoán rủi ro
2. **Trang giải thích mô hình**: Tìm hiểu cách mô hình đưa ra quyết định
3. **Trang giới thiệu**: Thông tin về dự án và dữ liệu

## Tác giả

- Data Engineer: Xử lý dữ liệu, làm sạch, xử lý giá trị thiếu
- ML Engineer: Huấn luyện mô hình, tối ưu và kiểm định hiệu năng
- API Developer: Xây dựng REST API, giao diện Streamlit
- DevOps Engineer: Xây dựng Docker, triển khai Docker Swarm
- Analyst & Tester: Kiểm thử, đánh giá kết quả, diễn giải mô hình

## Giấy phép

[MIT License](LICENSE) 