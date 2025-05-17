"""
Image processing utilities for enhancing and transforming images.
"""

import cv2
import numpy as np

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
def increase_brightness_hsv(image_hsv, value=30):
    """Tăng độ sáng của ảnh trong không gian màu HSV
    
    Args:
        image_hsv: Ảnh đầu vào trong không gian màu HSV
        value: Giá trị tăng độ sáng, mặc định là 30
        
    Returns:
        Ảnh HSV đã tăng độ sáng
    """
    # Tạo bản sao để không ảnh hưởng đến ảnh gốc
    result = image_hsv.copy()
    # Tăng giá trị kênh V (Value/Brightness)
    result[:, :, 2] = np.clip(result[:, :, 2] + value, 0, 255)
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
