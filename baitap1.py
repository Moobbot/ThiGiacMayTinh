import cv2
import numpy as np
import os
import logging
import time
from datetime import datetime

# Thiết lập logging
def setup_logging():
    """Thiết lập logging để ghi lại quá trình thực thi"""
    log_filename = f"baitap1_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Tạo logger
    logger = logging.getLogger('baitap1')
    logger.setLevel(logging.INFO)

    # Tạo file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Tạo console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Tạo formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Thêm handlers vào logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Hàm tiện ích
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

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def display_image(image, window_name, wait_time=1000):
    """Hiển thị ảnh trong cửa sổ

    Args:
        image: Ảnh cần hiển thị
        window_name: Tên cửa sổ
        wait_time: Thời gian chờ (ms) trước khi đóng cửa sổ, mặc định là 1000ms (1 giây)
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    # Đóng cửa sổ sau khi hiển thị để tránh quá nhiều cửa sổ mở cùng lúc
    cv2.destroyWindow(window_name)

# Các hàm xử lý ảnh RGB
def increase_brightness_rgb(image, value=30):
    """Tăng độ sáng của ảnh trong không gian màu RGB"""
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image.copy().astype(np.float32)
    # Tăng giá trị của tất cả các kênh màu
    result += value
    # Đảm bảo giá trị pixel nằm trong khoảng [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def negative_transform_rgb(image):
    """Biến đổi âm bản của ảnh"""
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = 255 - image.copy()
    return result

# Các hàm xử lý ảnh HSV
def increase_brightness_hsv(image_hsv, value=30):
    """Tăng độ sáng của ảnh trong không gian màu HSV"""
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image_hsv.copy()
    # Tăng giá trị kênh V (Value/Brightness)
    result[:, :, 2] = np.clip(result[:, :, 2] + value, 0, 255)
    return result

def increase_saturation_hsv(image_hsv, value=30):
    """Tăng độ bão hòa của ảnh trong không gian màu HSV"""
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image_hsv.copy()
    # Tăng giá trị kênh S (Saturation)
    result[:, :, 1] = np.clip(result[:, :, 1] + value, 0, 255)
    return result

# Hàm cân bằng histogram
def histogram_equalization(gray_image):
    """Cân bằng histogram cho ảnh xám"""
    # Tính histogram
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

    # Tính CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()

    # Chuẩn hóa CDF
    cdf_normalized = cdf * 255 / cdf[-1]

    # Áp dụng cân bằng histogram
    result = np.interp(gray_image.flatten(), bins[:-1], cdf_normalized)
    result = result.reshape(gray_image.shape).astype(np.uint8)

    return result

# Các hàm lọc ảnh
def mean_filter(image, kernel_size=3):
    """Lọc trung bình cho ảnh xám"""
    # Tạo kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # Lấy kích thước ảnh
    height, width = image.shape

    # Tạo ảnh kết quả
    result = np.zeros_like(image)

    # Padding ảnh
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')

    # Áp dụng lọc (chỉ xử lý một phần ảnh để tăng tốc độ)
    # Lấy mẫu 1/4 ảnh (mỗi 2 pixel theo chiều ngang và dọc)
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # Lấy vùng ảnh tương ứng với kernel
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Tính giá trị trung bình
            result[i, j] = np.sum(region * kernel)

            # Sao chép giá trị cho các pixel lân cận để tăng tốc độ
            if i+1 < height:
                result[i+1, j] = result[i, j]
            if j+1 < width:
                result[i, j+1] = result[i, j]
            if i+1 < height and j+1 < width:
                result[i+1, j+1] = result[i, j]

    return result.astype(np.uint8)

def median_filter(image, kernel_size=3):
    """Lọc trung vị cho ảnh xám"""
    # Lấy kích thước ảnh
    height, width = image.shape

    # Tạo ảnh kết quả
    result = np.zeros_like(image)

    # Padding ảnh
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')

    # Áp dụng lọc (chỉ xử lý một phần ảnh để tăng tốc độ)
    # Lấy mẫu 1/4 ảnh (mỗi 2 pixel theo chiều ngang và dọc)
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # Lấy vùng ảnh tương ứng với kernel
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Tính giá trị trung vị
            result[i, j] = np.median(region)

            # Sao chép giá trị cho các pixel lân cận để tăng tốc độ
            if i+1 < height:
                result[i+1, j] = result[i, j]
            if j+1 < width:
                result[i, j+1] = result[i, j]
            if i+1 < height and j+1 < width:
                result[i+1, j+1] = result[i, j]

    return result.astype(np.uint8)

# Hàm lưu kết quả
def save_results(images_dict, output_dir='output'):
    """Lưu các ảnh kết quả vào thư mục output và trả về danh sách các tệp đã lưu

    Args:
        images_dict: Dictionary chứa tên file và ảnh tương ứng
        output_dir: Thư mục đầu ra, mặc định là 'output'

    Returns:
        Danh sách đường dẫn đầy đủ của các file đã lưu
    """
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
    """Hàm chính của chương trình

    Args:
        show_images: Nếu True, hiển thị ảnh trong quá trình xử lý.
                    Nếu False, chỉ xử lý và lưu ảnh mà không hiển thị.
    """
    # Thiết lập logging
    logger = setup_logging()

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

    saved_files = save_results(images_to_save)
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

    print("Chương trình đã hoàn thành. Xem kết quả trong tệp log và các tệp ảnh đã tạo.")