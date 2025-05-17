"""
Logging utilities for the application.
"""

import logging
from datetime import datetime
import os

def setup_logging(log_dir='logs'):
    """Thiết lập logging để ghi lại quá trình thực thi
    
    Args:
        log_dir: Thư mục chứa các file log, mặc định là 'logs'
        
    Returns:
        Logger object đã được cấu hình
    """
    # Tạo thư mục logs nếu chưa tồn tại
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Tạo tên file log với timestamp
    log_filename = f"baitap1_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    # Tạo logger
    logger = logging.getLogger('baitap1')
    logger.setLevel(logging.INFO)
    
    # Xóa các handler cũ nếu có
    if logger.handlers:
        logger.handlers.clear()
    
    # Tạo file handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
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
