"""
Image I/O utilities for reading, writing, and displaying images.
"""

import cv2
import os

def resize_image_for_display(image, max_width=800, max_height=600):
    """Resize image to fit within specified dimensions while maintaining aspect ratio
    
    Args:
        image: Ảnh cần resize
        max_width: Chiều rộng tối đa
        max_height: Chiều cao tối đa
        
    Returns:
        Ảnh đã được resize
    """
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
    """Liệt kê các tệp đã lưu và kích thước của chúng
    
    Args:
        saved_files: Danh sách đường dẫn các file đã lưu
        
    Returns:
        Danh sách các chuỗi mô tả file và kích thước
    """
    result = []
    
    for file in saved_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024  # Kích thước tệp tính bằng KB
            result.append(f"  - {file} ({file_size:.2f} KB)")
        else:
            result.append(f"  - {file} (không tồn tại)")
    
    return result
