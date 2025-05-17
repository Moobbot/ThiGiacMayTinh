"""
Image processing utilities for enhancing and transforming images.
"""

import cv2
import numpy as np

# Hàm chuyển đổi ảnh màu sang ảnh xám
def bgr_to_gray(image):
    """Chuyển đổi ảnh BGR sang ảnh xám không sử dụng hàm cv2.cvtColor

    Công thức: Gray = 0.299*R + 0.587*G + 0.114*B

    Args:
        image: Ảnh đầu vào (BGR)

    Returns:
        Ảnh xám
    """
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    # Trích xuất các kênh màu B, G, R
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    # Áp dụng công thức chuyển đổi sang ảnh xám
    # Gray = 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Chuyển về kiểu dữ liệu uint8
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray

def compare_grayscale_implementations(custom_gray, opencv_gray):
    """So sánh kết quả giữa hàm chuyển đổi ảnh xám tự cài đặt và OpenCV

    Args:
        custom_gray: Ảnh xám từ hàm tự cài đặt
        opencv_gray: Ảnh xám từ hàm OpenCV

    Returns:
        Tuple chứa (ảnh hiển thị sự khác biệt, giá trị sai số trung bình)
    """
    # Tính sự khác biệt tuyệt đối giữa hai ảnh
    diff = cv2.absdiff(custom_gray, opencv_gray)

    # Tính sai số trung bình
    mean_error = np.mean(diff)

    # Tạo ảnh hiển thị sự khác biệt (nhân với hệ số để dễ nhìn)
    # Nhân với 10 để làm nổi bật sự khác biệt
    diff_display = np.clip(diff * 10, 0, 255).astype(np.uint8)

    return diff_display, mean_error

# Các hàm xử lý ảnh RGB
def increase_brightness_rgb(image, value=30):
    """Tăng độ sáng của ảnh trong không gian màu RGB

    Args:
        image: Ảnh đầu vào (BGR hoặc RGB)
        value: Giá trị tăng độ sáng, mặc định là 30

    Returns:
        Ảnh đã tăng độ sáng
    """
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image.copy().astype(np.float32)
    # Tăng giá trị của tất cả các kênh màu
    result += value
    # Đảm bảo giá trị pixel nằm trong khoảng [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def negative_transform_rgb(image):
    """Biến đổi âm bản của ảnh

    Args:
        image: Ảnh đầu vào (BGR hoặc RGB)

    Returns:
        Ảnh âm bản
    """
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = 255 - image.copy()
    return result

# Các hàm xử lý ảnh HSV
def increase_brightness_hsv(image_hsv, value=30, gamma=1.0, adaptive=False, protect_highlights=False, preserve_colors=False):
    """Tăng độ sáng của ảnh trong không gian màu HSV với nhiều tùy chọn nâng cao

    Args:
        image_hsv: Ảnh đầu vào trong không gian màu HSV
        value: Giá trị tăng độ sáng, mặc định là 30
        gamma: Hệ số gamma để điều chỉnh độ sáng phi tuyến (gamma < 1: tăng sáng vùng tối,
               gamma > 1: tăng sáng vùng sáng), mặc định là 1.0 (không áp dụng)
        adaptive: Nếu True, điều chỉnh độ sáng dựa trên histogram của ảnh, mặc định là False
        protect_highlights: Nếu True, bảo vệ các vùng sáng để tránh quá sáng, mặc định là False
        preserve_colors: Nếu True, giữ nguyên màu sắc khi tăng độ sáng, mặc định là False

    Returns:
        Ảnh HSV đã tăng độ sáng
    """
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image_hsv.copy()

    # Trích xuất kênh V (Value/Brightness)
    v_channel = result[:, :, 2].astype(np.float32)

    if adaptive:
        # Tính giá trị trung bình của kênh V
        mean_v = np.mean(v_channel)
        # Điều chỉnh giá trị tăng sáng dựa trên độ sáng trung bình
        # Nếu ảnh tối, tăng nhiều hơn; nếu ảnh sáng, tăng ít hơn
        adaptive_value = value * (1.0 - mean_v / 255.0) * 2.0
        value = max(min(adaptive_value, value * 2), value / 2)

    if gamma != 1.0:
        # Chuẩn hóa kênh V về khoảng [0, 1]
        normalized_v = v_channel / 255.0
        # Áp dụng hiệu chỉnh gamma
        gamma_corrected = np.power(normalized_v, 1.0/gamma)
        # Chuyển lại về khoảng [0, 255]
        v_channel = gamma_corrected * 255.0

    # Tăng độ sáng
    if protect_highlights:
        # Tạo mặt nạ cho các vùng sáng (giá trị V cao)
        highlight_mask = v_channel > 200
        # Tạo hệ số giảm dần cho các vùng sáng
        highlight_factor = np.ones_like(v_channel)
        highlight_factor[highlight_mask] = 1.0 - (v_channel[highlight_mask] - 200) / 55.0
        highlight_factor = np.clip(highlight_factor, 0.0, 1.0)
        # Áp dụng hệ số giảm dần cho giá trị tăng sáng
        v_channel = v_channel + value * highlight_factor
    else:
        # Tăng độ sáng đều cho tất cả các pixel
        v_channel = v_channel + value

    # Đảm bảo giá trị nằm trong khoảng [0, 255]
    v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)

    # Cập nhật kênh V
    result[:, :, 2] = v_channel

    # Nếu preserve_colors=True, điều chỉnh kênh S để giữ nguyên màu sắc
    if preserve_colors:
        # Khi tăng độ sáng, màu sắc có thể bị nhạt đi
        # Tăng độ bão hòa tỷ lệ với mức tăng độ sáng để bù đắp
        s_channel = result[:, :, 1].astype(np.float32)
        # Tính hệ số tăng độ bão hòa dựa trên mức tăng độ sáng
        saturation_factor = 1.0 + (value / 255.0) * 0.5
        # Áp dụng hệ số tăng độ bão hòa
        s_channel = s_channel * saturation_factor
        # Đảm bảo giá trị nằm trong khoảng [0, 255]
        result[:, :, 1] = np.clip(s_channel, 0, 255).astype(np.uint8)

    return result

def increase_saturation_hsv(image_hsv, value=30):
    """Tăng độ bão hòa của ảnh trong không gian màu HSV

    Args:
        image_hsv: Ảnh đầu vào trong không gian màu HSV
        value: Giá trị tăng độ bão hòa, mặc định là 30

    Returns:
        Ảnh HSV đã tăng độ bão hòa
    """
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image_hsv.copy()
    # Tăng giá trị kênh S (Saturation)
    result[:, :, 1] = np.clip(result[:, :, 1] + value, 0, 255)
    return result

# Hàm cân bằng histogram
def histogram_equalization(gray_image):
    """Cân bằng histogram cho ảnh xám

    Args:
        gray_image: Ảnh xám đầu vào

    Returns:
        Ảnh xám đã cân bằng histogram
    """
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
