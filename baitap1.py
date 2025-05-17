"""
Bài tập 1: Xử lý ảnh cơ bản
"""

import cv2
import os
import time
import argparse

# Import các module từ package utils
from utils.logger import setup_logging
from utils.image_io import (
    resize_image_for_display,
    display_image,
    save_results,
    list_saved_files
)
from utils.image_processing import (
    increase_brightness_rgb,
    negative_transform_rgb,
    increase_brightness_hsv,
    increase_saturation_hsv,
    histogram_equalization
)
from utils.filters import mean_filter, median_filter

def main(show_images=True):
    """Hàm chính của chương trình

    Args:
        show_images: Nếu True, hiển thị ảnh trong quá trình xử lý.
                    Nếu False, chỉ xử lý và lưu ảnh mà không hiển thị.
    """
    # Thiết lập logging với các tham số mặc định
    logger = setup_logging(log_dir='logs', logger_name='baitap1', log_prefix='baitap1_log')

    start_time = time.time()
    logger.info("Bắt đầu chương trình xử lý ảnh")

    # Hàm hiển thị ảnh tùy theo tham số show_images
    def show_image(img, title):
        if show_images:
            display_image(img, title)
        else:
            logger.info(f"Bỏ qua hiển thị: {title}")

    # Kiểm tra thư mục hiện tại
    logger.info(f"Thư mục làm việc hiện tại: {os.getcwd()}")
    logger.info(f"Các tệp trong thư mục: {os.listdir()}")

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
    show_image(img, '1. Ảnh gốc')

    # 2. Chuyển sang ảnh xám
    logger.info("Đang chuyển đổi sang ảnh xám...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(img_gray, '2. Ảnh xám')

    # 3. Xử lý ảnh trong không gian màu RGB
    logger.info("Đang áp dụng các kỹ thuật xử lý ảnh trong không gian màu RGB...")

    # Tăng độ sáng
    img_bright = increase_brightness_rgb(img, 50)
    show_image(img_bright, '3a. Tăng độ sáng (RGB)')

    # Biến đổi âm bản
    img_negative = negative_transform_rgb(img)
    show_image(img_negative, '3b. Âm bản (RGB)')

    # 4. Xử lý ảnh trong không gian màu HSV
    logger.info("Đang áp dụng các kỹ thuật xử lý ảnh trong không gian màu HSV...")

    # Chuyển đổi sang HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Tăng độ sáng HSV
    img_bright_hsv = increase_brightness_hsv(img_hsv, 50)
    img_bright_hsv_bgr = cv2.cvtColor(img_bright_hsv, cv2.COLOR_HSV2BGR)
    show_image(img_bright_hsv_bgr, '4a. Tăng độ sáng (HSV)')

    # Tăng độ bão hòa HSV
    img_saturated_hsv = increase_saturation_hsv(img_hsv, 50)
    img_saturated_hsv_bgr = cv2.cvtColor(img_saturated_hsv, cv2.COLOR_HSV2BGR)
    show_image(img_saturated_hsv_bgr, '4b. Tăng độ bão hòa (HSV)')

    # 5. Cân bằng histogram
    logger.info("Đang thực hiện cân bằng histogram...")
    img_equalized = histogram_equalization(img_gray)
    show_image(img_equalized, '5. Cân bằng histogram')

    # 6. Lọc ảnh
    logger.info("Đang áp dụng bộ lọc trung bình...")
    img_mean_filtered = mean_filter(img_gray, 3)
    show_image(img_mean_filtered, '6a. Lọc trung bình')

    logger.info("Đang áp dụng bộ lọc trung vị...")
    img_median_filtered = median_filter(img_gray, 3)
    show_image(img_median_filtered, '6b. Lọc trung vị')

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

    saved_files = save_results(images_to_save, output_dir='output')
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
    import argparse

    # Tạo parser để xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Chương trình xử lý ảnh')
    parser.add_argument('--no-display', action='store_true',
                        help='Không hiển thị ảnh trong quá trình xử lý')

    args = parser.parse_args()

    # Gọi hàm main với tham số show_images phù hợp
    main(show_images=not args.no_display)

    print("Chương trình đã hoàn thành. Xem kết quả trong thư mục output và logs.")