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
   pip install opencv-python numpy
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

Kết quả sẽ được lưu trong thư mục `output` và log được lưu trong thư mục `logs`.

## Cấu trúc dự án

```plaintext
BaiTap_TGMT/
├── baitap1.py           # File chính
├── img_minhhoa.jpg      # Ảnh đầu vào
├── logs/                # Thư mục chứa các file log
├── output/              # Thư mục chứa các ảnh kết quả
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

## Tác giả

Dự án được phát triển bởi **NgoTam** cho môn học Thị giác máy tính.
