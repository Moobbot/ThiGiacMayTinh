"""
So sánh kết quả giữa phương pháp tự viết và phương pháp sử dụng hàm có sẵn của OpenCV
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def ensure_directory_exists(directory):
    """Đảm bảo thư mục tồn tại, nếu không thì tạo mới"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images(custom_dir, opencv_dir):
    """Tải các cặp ảnh từ hai thư mục để so sánh"""
    image_pairs = {}
    
    # Kiểm tra xem các thư mục có tồn tại không
    if not os.path.exists(custom_dir):
        print(f"Thư mục {custom_dir} không tồn tại!")
        return image_pairs
    
    if not os.path.exists(opencv_dir):
        print(f"Thư mục {opencv_dir} không tồn tại!")
        return image_pairs
    
    # Lấy danh sách các file trong thư mục custom
    custom_files = [f for f in os.listdir(custom_dir) if f.endswith('.jpg')]
    
    # Tìm các file tương ứng trong thư mục opencv
    for file in custom_files:
        opencv_file = os.path.join(opencv_dir, file)
        custom_file = os.path.join(custom_dir, file)
        
        if os.path.exists(opencv_file):
            # Đọc cả hai ảnh
            custom_img = cv2.imread(custom_file)
            opencv_img = cv2.imread(opencv_file)
            
            # Đảm bảo cả hai ảnh có cùng kích thước
            if custom_img.shape != opencv_img.shape:
                print(f"Kích thước không khớp cho {file}. Đang resize...")
                opencv_img = cv2.resize(opencv_img, (custom_img.shape[1], custom_img.shape[0]))
            
            # Thêm vào dictionary
            image_pairs[file] = (custom_img, opencv_img)
    
    return image_pairs

def calculate_metrics(img1, img2):
    """Tính toán các chỉ số so sánh giữa hai ảnh"""
    # Chuyển sang ảnh xám nếu là ảnh màu
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Tính SSIM (Structural Similarity Index)
    ssim_value = ssim(img1_gray, img2_gray)
    
    # Tính PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = psnr(img1_gray, img2_gray)
    
    # Tính MSE (Mean Squared Error)
    mse = np.mean((img1_gray.astype(np.float64) - img2_gray.astype(np.float64)) ** 2)
    
    # Tính MAE (Mean Absolute Error)
    mae = np.mean(np.abs(img1_gray.astype(np.float64) - img2_gray.astype(np.float64)))
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'mse': mse,
        'mae': mae
    }

def create_comparison_image(img1, img2, title1, title2, metrics):
    """Tạo ảnh so sánh giữa hai ảnh"""
    # Tạo ảnh sai biệt (difference image)
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    diff = cv2.absdiff(img1_gray, img2_gray)
    # Tăng độ tương phản của ảnh sai biệt để dễ nhìn
    diff = cv2.convertScaleAbs(diff, alpha=5)
    
    # Tạo hình vẽ
    plt.figure(figsize=(15, 5))
    
    # Hiển thị ảnh 1
    plt.subplot(1, 3, 1)
    if len(img1.shape) == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    
    # Hiển thị ảnh 2
    plt.subplot(1, 3, 2)
    if len(img2.shape) == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    # Hiển thị ảnh sai biệt
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='hot')
    plt.title(f'Sai biệt\nSSIM: {metrics["ssim"]:.4f}, PSNR: {metrics["psnr"]:.2f}dB\nMSE: {metrics["mse"]:.2f}, MAE: {metrics["mae"]:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    """Hàm chính"""
    # Thư mục chứa ảnh
    custom_dir = 'output'
    opencv_dir = 'output_opencv'
    comparison_dir = 'comparison'
    
    # Đảm bảo thư mục comparison tồn tại
    ensure_directory_exists(comparison_dir)
    
    # Tải các cặp ảnh
    print("Đang tải các cặp ảnh để so sánh...")
    image_pairs = load_images(custom_dir, opencv_dir)
    
    if not image_pairs:
        print("Không tìm thấy cặp ảnh nào để so sánh!")
        return
    
    print(f"Đã tìm thấy {len(image_pairs)} cặp ảnh để so sánh.")
    
    # Tạo bảng so sánh
    comparison_table = []
    
    # So sánh từng cặp ảnh
    for filename, (custom_img, opencv_img) in image_pairs.items():
        print(f"Đang so sánh {filename}...")
        
        # Tính các chỉ số so sánh
        metrics = calculate_metrics(custom_img, opencv_img)
        comparison_table.append({
            'filename': filename,
            **metrics
        })
        
        # Tạo ảnh so sánh
        fig = create_comparison_image(
            custom_img, opencv_img,
            f"Tự viết: {filename}", 
            f"OpenCV: {filename}",
            metrics
        )
        
        # Lưu ảnh so sánh
        comparison_path = os.path.join(comparison_dir, f"compare_{filename}")
        fig.savefig(comparison_path, dpi=150)
        plt.close(fig)
        
        print(f"  SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB")
        print(f"  MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}")
    
    # Tạo bảng tổng hợp
    print("\nBảng tổng hợp kết quả so sánh:")
    print("-" * 80)
    print(f"{'Tên file':<25} {'SSIM':<10} {'PSNR (dB)':<12} {'MSE':<10} {'MAE':<10}")
    print("-" * 80)
    
    for item in comparison_table:
        print(f"{item['filename']:<25} {item['ssim']:<10.4f} {item['psnr']:<12.2f} {item['mse']:<10.2f} {item['mae']:<10.2f}")
    
    print("-" * 80)
    
    # Tính giá trị trung bình
    avg_ssim = np.mean([item['ssim'] for item in comparison_table])
    avg_psnr = np.mean([item['psnr'] for item in comparison_table])
    avg_mse = np.mean([item['mse'] for item in comparison_table])
    avg_mae = np.mean([item['mae'] for item in comparison_table])
    
    print(f"{'Trung bình':<25} {avg_ssim:<10.4f} {avg_psnr:<12.2f} {avg_mse:<10.2f} {avg_mae:<10.2f}")
    
    # Lưu bảng tổng hợp vào file
    with open(os.path.join(comparison_dir, 'comparison_results.txt'), 'w', encoding='utf-8') as f:
        f.write("Bảng tổng hợp kết quả so sánh:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Tên file':<25} {'SSIM':<10} {'PSNR (dB)':<12} {'MSE':<10} {'MAE':<10}\n")
        f.write("-" * 80 + "\n")
        
        for item in comparison_table:
            f.write(f"{item['filename']:<25} {item['ssim']:<10.4f} {item['psnr']:<12.2f} {item['mse']:<10.2f} {item['mae']:<10.2f}\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'Trung bình':<25} {avg_ssim:<10.4f} {avg_psnr:<12.2f} {avg_mse:<10.2f} {avg_mae:<10.2f}\n")
    
    print(f"\nKết quả so sánh đã được lưu vào thư mục {comparison_dir}")
    print(f"Các ảnh so sánh đã được lưu với tiền tố 'compare_'")
    print(f"Bảng tổng hợp đã được lưu vào file 'comparison_results.txt'")

if __name__ == "__main__":
    main()
