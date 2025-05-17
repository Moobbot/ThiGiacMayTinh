"""
Image filtering utilities.
"""

import numpy as np

def mean_filter(image, kernel_size=3):
    """Lọc trung bình cho ảnh xám
    
    Args:
        image: Ảnh xám đầu vào
        kernel_size: Kích thước kernel, mặc định là 3
        
    Returns:
        Ảnh đã lọc trung bình
    """
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
    """Lọc trung vị cho ảnh xám
    
    Args:
        image: Ảnh xám đầu vào
        kernel_size: Kích thước kernel, mặc định là 3
        
    Returns:
        Ảnh đã lọc trung vị
    """
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
