# Bài tập xử lý ảnh cơ bản với OpenCV

Dự án này thực hiện các kỹ thuật xử lý ảnh cơ bản sử dụng thư viện OpenCV và Python. Các thuật toán xử lý ảnh được viết từ đầu (không sử dụng các hàm có sẵn của OpenCV) để hiểu rõ nguyên lý hoạt động.

## Mục lục

- [Bài tập xử lý ảnh cơ bản với OpenCV](#bài-tập-xử-lý-ảnh-cơ-bản-với-opencv)
  - [Mục lục](#mục-lục)
  - [Tính năng](#tính-năng)
  - [Cài đặt](#cài-đặt)
  - [Cách sử dụng](#cách-sử-dụng)
  - [Cấu trúc dự án](#cấu-trúc-dự-án)
  - [Các kỹ thuật xử lý ảnh](#các-kỹ-thuật-xử-lý-ảnh)
  - [Kết quả](#kết-quả)
  - [So sánh với OpenCV](#so-sánh-với-opencv)
    - [Kết quả so sánh](#kết-quả-so-sánh)
    - [Nhận xét](#nhận-xét)
  - [Tác giả](#tác-giả)

## Tính năng

Chương trình thực hiện các kỹ thuật xử lý ảnh cơ bản sau:

1. Đọc và hiển thị ảnh
2. Chuyển đổi sang ảnh xám
3. Xử lý ảnh trong không gian màu RGB:
   - Tăng độ sáng
   - Biến đổi âm bản
4. Xử lý ảnh trong không gian màu HSV:
   - Tăng độ sáng (thông qua kênh V)
   - Tăng độ bão hòa (thông qua kênh S)
5. Cân bằng histogram
6. Lọc ảnh:
   - Lọc trung bình
   - Lọc trung vị

Tất cả các thuật toán xử lý ảnh đều được viết từ đầu, không sử dụng các hàm có sẵn của OpenCV.

## Cài đặt

1. Clone repository:

   ```bash
   git clone <repository-url>
   cd BaiTap_TGMT
   ```

2. Cài đặt các thư viện cần thiết:

   ```bash
   pip install -r requirements.txt
   ```

   Hoặc cài đặt thủ công:

   ```bash
   pip install opencv-python numpy matplotlib scikit-image
   ```

## Cách sử dụng

Chạy chương trình với lệnh:

```bash
python baitap1.py
```

Tùy chọn:

- `--no-display`: Chạy chương trình mà không hiển thị ảnh (chỉ xử lý và lưu kết quả)

```bash
python baitap1.py --no-display
```

Để chạy phiên bản sử dụng các hàm có sẵn của OpenCV:

```bash
python baitap1_opencv.py --no-display
```

Để so sánh kết quả giữa hai phương pháp:

```bash
python compare_results.py
```

Kết quả sẽ được lưu trong các thư mục sau:

- `output`: Kết quả từ phương pháp tự viết
- `output_opencv`: Kết quả từ phương pháp sử dụng hàm có sẵn của OpenCV
- `comparison`: Kết quả so sánh giữa hai phương pháp
- `logs`: Các file log

## Cấu trúc dự án

```plaintext
BaiTap_TGMT/
├── baitap1.py           # File chính (phương pháp tự viết)
├── baitap1_opencv.py    # File sử dụng hàm có sẵn của OpenCV
├── compare_results.py   # Script so sánh kết quả
├── requirements.txt     # Danh sách các thư viện cần thiết
├── img_minhhoa.jpg      # Ảnh đầu vào
├── logs/                # Thư mục chứa các file log
├── output/              # Thư mục chứa kết quả từ phương pháp tự viết
├── output_opencv/       # Thư mục chứa kết quả từ phương pháp OpenCV
├── comparison/          # Thư mục chứa kết quả so sánh
└── utils/               # Package chứa các module tiện ích
    ├── __init__.py
    ├── filters.py       # Các hàm lọc ảnh
    ├── image_io.py      # Các hàm đọc/ghi ảnh
    ├── image_processing.py  # Các hàm xử lý ảnh
    └── logger.py        # Các hàm thiết lập logging
```

## Các kỹ thuật xử lý ảnh

1. **Tăng độ sáng (RGB)**: Tăng giá trị của tất cả các kênh màu R, G, B.
2. **Biến đổi âm bản (RGB)**: Đảo ngược giá trị của tất cả các pixel (255 - giá trị pixel).
3. **Tăng độ sáng (HSV)**: Tăng giá trị của kênh V (Value) trong không gian màu HSV.
4. **Tăng độ bão hòa (HSV)**: Tăng giá trị của kênh S (Saturation) trong không gian màu HSV.
5. **Cân bằng histogram**: Cải thiện độ tương phản của ảnh bằng cách phân phối đều các giá trị pixel.
6. **Lọc trung bình**: Làm mịn ảnh bằng cách thay thế mỗi pixel bằng giá trị trung bình của các pixel lân cận.
7. **Lọc trung vị**: Loại bỏ nhiễu bằng cách thay thế mỗi pixel bằng giá trị trung vị của các pixel lân cận.

## Kết quả

Các ảnh kết quả được lưu trong thư mục `output` với các tên file sau:

- `1_anh_goc.jpg`: Ảnh gốc
- `2_anh_xam.jpg`: Ảnh xám
- `3a_tang_do_sang_rgb.jpg`: Ảnh đã tăng độ sáng trong không gian màu RGB
- `3b_am_ban_rgb.jpg`: Ảnh âm bản trong không gian màu RGB
- `4a_tang_do_sang_hsv.jpg`: Ảnh đã tăng độ sáng trong không gian màu HSV
- `4b_tang_do_bao_hoa_hsv.jpg`: Ảnh đã tăng độ bão hòa trong không gian màu HSV
- `5_can_bang_histogram.jpg`: Ảnh đã cân bằng histogram
- `6a_loc_trung_binh.jpg`: Ảnh đã lọc trung bình
- `6b_loc_trung_vi.jpg`: Ảnh đã lọc trung vị

## So sánh với OpenCV

Dự án này cũng bao gồm một phiên bản sử dụng các hàm có sẵn của OpenCV (`baitap1_opencv.py`) và một script để so sánh kết quả giữa hai phương pháp (`compare_results.py`).

### Kết quả so sánh

Kết quả so sánh giữa phương pháp tự viết và phương pháp sử dụng hàm có sẵn của OpenCV:

| Tên file | SSIM | PSNR (dB) | MSE | MAE |
|----------|------|-----------|-----|-----|
| 1_anh_goc.jpg | 1.0000 | ∞ | 0.00 | 0.00 |
| 2_anh_xam.jpg | 1.0000 | ∞ | 0.00 | 0.00 |
| 3a_tang_do_sang_rgb.jpg | 1.0000 | ∞ | 0.00 | 0.00 |
| 3b_am_ban_rgb.jpg | 1.0000 | ∞ | 0.00 | 0.00 |
| 4a_tang_do_sang_hsv.jpg | 0.6229 | 10.65 | 5604.29 | 35.19 |
| 4b_tang_do_bao_hoa_hsv.jpg | 0.8601 | 19.46 | 736.03 | 9.85 |
| 5_can_bang_histogram.jpg | 0.9968 | 44.80 | 2.16 | 1.06 |
| 6a_loc_trung_binh.jpg | 0.8934 | 31.77 | 43.28 | 3.94 |
| 6b_loc_trung_vi.jpg | 0.8537 | 29.56 | 71.96 | 4.50 |

Trong đó:

- **SSIM (Structural Similarity Index)**: Đo lường sự tương đồng về cấu trúc giữa hai ảnh (1.0 = giống hệt nhau)
- **PSNR (Peak Signal-to-Noise Ratio)**: Đo lường chất lượng tái tạo (càng cao càng tốt)
- **MSE (Mean Squared Error)**: Sai số bình phương trung bình (càng thấp càng tốt)
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình (càng thấp càng tốt)

### Nhận xét

1. **Các phép biến đổi RGB cơ bản** (ảnh gốc, ảnh xám, tăng độ sáng RGB, âm bản RGB) có kết quả giống hệt nhau giữa hai phương pháp (SSIM = 1.0).

2. **Các phép biến đổi HSV** có sự khác biệt đáng kể:
   - Tăng độ sáng HSV: SSIM = 0.6229, PSNR = 10.65dB (khác biệt lớn)
   - Tăng độ bão hòa HSV: SSIM = 0.8601, PSNR = 19.46dB (khác biệt vừa phải)

3. **Cân bằng histogram** có kết quả gần giống nhau (SSIM = 0.9968, PSNR = 44.80dB).

4. **Lọc ảnh** có sự khác biệt nhỏ:
   - Lọc trung bình: SSIM = 0.8934, PSNR = 31.77dB
   - Lọc trung vị: SSIM = 0.8537, PSNR = 29.56dB

5. **Hiệu suất**: Phiên bản sử dụng hàm có sẵn của OpenCV chạy nhanh hơn nhiều (0.14 giây) so với phiên bản tự viết (3.36 giây).

Các ảnh so sánh chi tiết được lưu trong thư mục `comparison` với tiền tố `compare_`.

## Tác giả

Dự án được phát triển bởi **NgoTam** cho môn học Thị giác máy tính.
