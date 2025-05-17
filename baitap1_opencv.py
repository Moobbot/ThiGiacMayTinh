"""
Bài tập xử lý ảnh cơ bản với OpenCV - Sử dụng các hàm có sẵn của OpenCV
Dùng để so sánh với phiên bản tự viết các thuật toán
"""

import cv2
import numpy as np
import os
import time
import argparse
import logging

# Import setup_logging từ utils.logger
from utils.logger import setup_logging

def resize_image_for_display(image, max_width=800, max_height=600):
    """Resize image to fit within specified dimensions while maintaining aspect ratio"""
    height, width = image.shape[:2]

    # Calculate the ratio of the width and height to the max dimensions
    ratio_width = max_width / width
    ratio_height = max_height / height

    # Use the smaller ratio to ensure the image fits within the specified dimensions
    ratio = min(ratio_width, ratio_height)

    # Calculate new dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def display_image(image, window_name, wait_time=1000):
    """Hiển thị ảnh trong cửa sổ"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    # Đóng cửa sổ sau khi hiển thị để tránh quá nhiều cửa sổ mở cùng lúc
    cv2.destroyWindow(window_name)

def save_results(images_dict, output_dir='output_opencv'):
    """Lưu các ảnh kết quả vào thư mục output và trả về danh sách các tệp đã lưu"""
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_files = []

    for filename, image in images_dict.items():
        # Tạo đường dẫn đầy đủ
        output_path = os.path.join(output_dir, filename)
        # Lưu ảnh
        cv2.imwrite(output_path, image)
        # Thêm vào danh sách kết quả
        saved_files.append(output_path)

    return saved_files

def list_saved_files(saved_files):
    """Liệt kê các tệp đã lưu và kích thước của chúng"""
    result = []

    for file in saved_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024  # Kích thước tệp tính bằng KB
            result.append(f"  - {file} ({file_size:.2f} KB)")
        else:
            result.append(f"  - {file} (không tồn tại)")

    return result

def main(show_images=True):
    """Hàm chính của chương trình"""
    # Thiết lập logging với tên logger và tiền tố file log tùy chỉnh
    logger = setup_logging(log_dir='logs', logger_name='opencv_compare', log_prefix='opencv_log')

    start_time = time.time()
    logger.info("Bắt đầu chương trình xử lý ảnh (sử dụng hàm có sẵn của OpenCV)")

    # Hàm hiển thị ảnh tùy theo tham số show_images
    def show_image(img, title):
        if show_images:
            display_image(img, title)
        else:
            logger.info(f"Bỏ qua hiển thị: {title}")

    # Kiểm tra thư mục hiện tại
    logger.info(f"Thư mục làm việc hiện tại: {os.getcwd()}")

    # 1. Đọc ảnh
    logger.info("Đang đọc ảnh...")
    img = cv2.imread('img_minhhoa.jpg')
    if img is None:
        logger.error("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
        return

    logger.info(f"Đọc ảnh thành công. Kích thước: {img.shape}")

    # Giảm kích thước ảnh để tăng tốc độ xử lý
    logger.info("Đang thay đổi kích thước ảnh để tăng tốc độ xử lý...")
    img = resize_image_for_display(img, max_width=800, max_height=600)
    logger.info(f"Kích thước ảnh sau khi thay đổi: {img.shape}")

    # Hiển thị ảnh gốc
    show_image(img, '1. Ảnh gốc (OpenCV)')

    # 2. Chuyển sang ảnh xám
    logger.info("Đang chuyển đổi sang ảnh xám...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(img_gray, '2. Ảnh xám (OpenCV)')

    # 3. Xử lý ảnh trong không gian màu RGB
    logger.info("Đang áp dụng các kỹ thuật xử lý ảnh trong không gian màu RGB...")

    # Tăng độ sáng sử dụng convertScaleAbs
    logger.info("Tăng độ sáng (RGB) sử dụng convertScaleAbs...")
    alpha = 1.0  # Độ tương phản
    beta = 50    # Độ sáng
    img_bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    show_image(img_bright, '3a. Tăng độ sáng (RGB) - OpenCV')

    # Biến đổi âm bản sử dụng bitwise_not
    logger.info("Biến đổi âm bản (RGB) sử dụng bitwise_not...")
    img_negative = cv2.bitwise_not(img)
    show_image(img_negative, '3b. Âm bản (RGB) - OpenCV')

    # 4. Xử lý ảnh trong không gian màu HSV
    logger.info("Đang áp dụng các kỹ thuật xử lý ảnh trong không gian màu HSV...")

    # Chuyển đổi sang HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Tăng độ sáng HSV (kênh V)
    logger.info("Tăng độ sáng (HSV) bằng cách tăng kênh V...")
    img_bright_hsv = img_hsv.copy()
    img_bright_hsv[:, :, 2] = cv2.add(img_bright_hsv[:, :, 2], 50)  # Tăng kênh V
    img_bright_hsv_bgr = cv2.cvtColor(img_bright_hsv, cv2.COLOR_HSV2BGR)
    show_image(img_bright_hsv_bgr, '4a. Tăng độ sáng (HSV) - OpenCV')

    # Tăng độ bão hòa HSV (kênh S)
    logger.info("Tăng độ bão hòa (HSV) bằng cách tăng kênh S...")
    img_saturated_hsv = img_hsv.copy()
    img_saturated_hsv[:, :, 1] = cv2.add(img_saturated_hsv[:, :, 1], 50)  # Tăng kênh S
    img_saturated_hsv_bgr = cv2.cvtColor(img_saturated_hsv, cv2.COLOR_HSV2BGR)
    show_image(img_saturated_hsv_bgr, '4b. Tăng độ bão hòa (HSV) - OpenCV')

    # 5. Cân bằng histogram
    logger.info("Đang thực hiện cân bằng histogram sử dụng equalizeHist...")
    img_equalized = cv2.equalizeHist(img_gray)
    show_image(img_equalized, '5. Cân bằng histogram - OpenCV')

    # 6. Lọc ảnh
    logger.info("Đang áp dụng bộ lọc trung bình sử dụng blur...")
    img_mean_filtered = cv2.blur(img_gray, (3, 3))  # Kernel 3x3
    show_image(img_mean_filtered, '6a. Lọc trung bình - OpenCV')

    logger.info("Đang áp dụng bộ lọc trung vị sử dụng medianBlur...")
    img_median_filtered = cv2.medianBlur(img_gray, 3)  # Kernel 3x3
    show_image(img_median_filtered, '6b. Lọc trung vị - OpenCV')

    # Lưu kết quả
    logger.info("Đang lưu kết quả...")
    images_to_save = {
        '1_anh_goc.jpg': img,
        '2_anh_xam.jpg': img_gray,
        '3a_tang_do_sang_rgb.jpg': img_bright,
        '3b_am_ban_rgb.jpg': img_negative,
        '4a_tang_do_sang_hsv.jpg': img_bright_hsv_bgr,
        '4b_tang_do_bao_hoa_hsv.jpg': img_saturated_hsv_bgr,
        '5_can_bang_histogram.jpg': img_equalized,
        '6a_loc_trung_binh.jpg': img_mean_filtered,
        '6b_loc_trung_vi.jpg': img_median_filtered
    }

    saved_files = save_results(images_to_save, output_dir='output_opencv')
    logger.info("Lưu kết quả thành công.")

    # Liệt kê các tệp đã lưu
    logger.info("Danh sách các tệp đã lưu:")
    for file_info in list_saved_files(saved_files):
        logger.info(file_info)

    # Tính thời gian thực thi
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Thời gian thực thi: {execution_time:.2f} giây")

    logger.info("Chương trình sẽ tự động đóng cửa sổ sau 3 giây...")
    # Đợi 3 giây rồi tự động đóng tất cả cửa sổ
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    logger.info("Chương trình hoàn thành.")

if __name__ == "__main__":
    # Tạo parser để xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Chương trình xử lý ảnh sử dụng hàm có sẵn của OpenCV')
    parser.add_argument('--no-display', action='store_true',
                        help='Không hiển thị ảnh trong quá trình xử lý')

    args = parser.parse_args()

    # Gọi hàm main với tham số show_images phù hợp
    main(show_images=not args.no_display)

    print("Chương trình đã hoàn thành. Xem kết quả trong thư mục output_opencv và logs.")
