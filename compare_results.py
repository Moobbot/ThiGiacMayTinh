"""
So sánh kết quả giữa phương pháp tự viết và phương pháp sử dụng hàm có sẵn của OpenCV
Tập trung vào so sánh chuyển đổi ảnh xám
Sử dụng ảnh với kích thước gốc (không resize) để đảm bảo độ chính xác của so sánh
"""

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import hàm tự cài đặt để so sánh trực tiếp
from utils.image_processing import bgr_to_gray

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

def create_three_way_comparison(original_img, custom_img, opencv_img, filename, comparison_dir):
    """Tạo ảnh so sánh ba chiều: ảnh gốc, ảnh từ phương pháp tự cài đặt, và ảnh từ OpenCV

    Args:
        original_img: Ảnh gốc trước khi xử lý
        custom_img: Ảnh từ phương pháp tự cài đặt
        opencv_img: Ảnh từ phương pháp OpenCV
        filename: Tên file ảnh
        comparison_dir: Thư mục lưu kết quả so sánh
    """
    # Tạo ảnh so sánh với matplotlib
    plt.figure(figsize=(15, 5))

    # Hiển thị ảnh gốc
    plt.subplot(1, 3, 1)
    if len(original_img.shape) == 3:
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original_img, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')

    # Hiển thị ảnh từ phương pháp tự cài đặt
    plt.subplot(1, 3, 2)
    if len(custom_img.shape) == 3:
        plt.imshow(cv2.cvtColor(custom_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(custom_img, cmap='gray')
    plt.title(f'Tự cài đặt: {filename}')
    plt.axis('off')

    # Hiển thị ảnh từ OpenCV
    plt.subplot(1, 3, 3)
    if len(opencv_img.shape) == 3:
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(opencv_img, cmap='gray')
    plt.title(f'OpenCV: {filename}')
    plt.axis('off')

    plt.tight_layout()

    # Lưu ảnh so sánh
    comparison_path = os.path.join(comparison_dir, f"three_way_{filename}")
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    return comparison_path

def create_detailed_comparison(img1, img2, filename, metrics, comparison_dir):
    """Tạo ảnh so sánh chi tiết giữa hai ảnh, tương tự như so sánh ảnh xám

    Args:
        img1: Ảnh từ phương pháp tự cài đặt
        img2: Ảnh từ phương pháp OpenCV
        filename: Tên file ảnh
        metrics: Dictionary chứa các chỉ số so sánh
        comparison_dir: Thư mục lưu kết quả so sánh
    """
    # Chuyển sang ảnh xám nếu là ảnh màu để tính toán sự khác biệt
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2

    # Tạo ảnh hiển thị sự khác biệt
    diff = cv2.absdiff(img1_gray, img2_gray)
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=10)  # Tăng độ tương phản để dễ nhìn
    cv2.imwrite(os.path.join(comparison_dir, f"diff_{filename}"), diff_enhanced)

    # Tạo ảnh so sánh chi tiết với matplotlib
    plt.figure(figsize=(15, 10))

    # Hiển thị ảnh gốc từ phương pháp tự cài đặt
    plt.subplot(2, 2, 1)
    if len(img1.shape) == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1, cmap='gray')
    plt.title(f'Tự cài đặt: {filename}')
    plt.axis('off')

    # Hiển thị ảnh từ OpenCV
    plt.subplot(2, 2, 2)
    if len(img2.shape) == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2, cmap='gray')
    plt.title(f'OpenCV: {filename}')
    plt.axis('off')

    # Hiển thị ảnh xám của cả hai phương pháp
    plt.subplot(2, 2, 3)
    plt.imshow(img1_gray, cmap='gray')
    plt.title('Ảnh xám (tự cài đặt)')
    plt.axis('off')

    # Hiển thị sự khác biệt
    plt.subplot(2, 2, 4)
    plt.imshow(diff_enhanced, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Sự khác biệt (x10)\nSSIM: {metrics["ssim"]:.4f}, PSNR: {metrics["psnr"]:.2f}dB\nMSE: {metrics["mse"]:.2f}, MAE: {metrics["mae"]:.2f}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"detailed_{filename}"), dpi=150)
    plt.close()

    # Tạo ảnh hiển thị sự khác biệt với colormap khác (heatmap)
    plt.figure(figsize=(10, 8))
    plt.imshow(diff_enhanced, cmap='jet')
    plt.colorbar(label='Độ khác biệt')
    plt.title(f'Sự khác biệt giữa hai phương pháp xử lý ảnh: {filename}\nSSIM: {metrics["ssim"]:.4f}, PSNR: {metrics["psnr"]:.2f}dB')
    plt.axis('off')
    plt.tight_layout()

    # Lưu ảnh sự khác biệt
    diff_path = os.path.join(comparison_dir, f"heatmap_{filename}")
    plt.savefig(diff_path, dpi=150)
    plt.close()

    # Nếu là ảnh màu, tạo thêm histogram cho từng kênh màu
    if len(img1.shape) == 3:
        # Tạo histogram cho từng kênh màu
        plt.figure(figsize=(15, 10))

        # Kênh B
        plt.subplot(3, 2, 1)
        plt.hist(img1[:,:,0].ravel(), 256, [0, 256], color='blue', alpha=0.7)
        plt.title('Histogram kênh B (tự cài đặt)')
        plt.xlabel('Giá trị pixel')
        plt.ylabel('Số lượng pixel')

        plt.subplot(3, 2, 2)
        plt.hist(img2[:,:,0].ravel(), 256, [0, 256], color='blue', alpha=0.7)
        plt.title('Histogram kênh B (OpenCV)')
        plt.xlabel('Giá trị pixel')
        plt.ylabel('Số lượng pixel')

        # Kênh G
        plt.subplot(3, 2, 3)
        plt.hist(img1[:,:,1].ravel(), 256, [0, 256], color='green', alpha=0.7)
        plt.title('Histogram kênh G (tự cài đặt)')
        plt.xlabel('Giá trị pixel')
        plt.ylabel('Số lượng pixel')

        plt.subplot(3, 2, 4)
        plt.hist(img2[:,:,1].ravel(), 256, [0, 256], color='green', alpha=0.7)
        plt.title('Histogram kênh G (OpenCV)')
        plt.xlabel('Giá trị pixel')
        plt.ylabel('Số lượng pixel')

        # Kênh R
        plt.subplot(3, 2, 5)
        plt.hist(img1[:,:,2].ravel(), 256, [0, 256], color='red', alpha=0.7)
        plt.title('Histogram kênh R (tự cài đặt)')
        plt.xlabel('Giá trị pixel')
        plt.ylabel('Số lượng pixel')

        plt.subplot(3, 2, 6)
        plt.hist(img2[:,:,2].ravel(), 256, [0, 256], color='red', alpha=0.7)
        plt.title('Histogram kênh R (OpenCV)')
        plt.xlabel('Giá trị pixel')
        plt.ylabel('Số lượng pixel')

        plt.tight_layout()

        # Lưu histogram
        hist_path = os.path.join(comparison_dir, f"histogram_{filename}")
        plt.savefig(hist_path, dpi=150)
        plt.close()
    else:
        # Tạo histogram cho ảnh xám
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(img1.ravel(), 256, [0, 256], color='blue', alpha=0.7)
        plt.title('Histogram (tự cài đặt)')
        plt.xlabel('Mức xám')
        plt.ylabel('Số lượng pixel')

        plt.subplot(1, 2, 2)
        plt.hist(img2.ravel(), 256, [0, 256], color='red', alpha=0.7)
        plt.title('Histogram (OpenCV)')
        plt.xlabel('Mức xám')
        plt.ylabel('Số lượng pixel')

        plt.tight_layout()

        # Lưu histogram
        hist_path = os.path.join(comparison_dir, f"histogram_{filename}")
        plt.savefig(hist_path, dpi=150)
        plt.close()

def compare_grayscale_from_files():
    """So sánh kết quả chuyển đổi ảnh xám từ các file đã lưu"""
    # Đường dẫn đến các file ảnh xám
    custom_gray_path = os.path.join('output', '2_anh_xam_tu_cai_dat.jpg')
    opencv_gray_path = os.path.join('output_opencv', '2_anh_xam.jpg')
    comparison_dir = 'comparison'

    # Đảm bảo thư mục comparison tồn tại
    ensure_directory_exists(comparison_dir)

    # Kiểm tra xem các file có tồn tại không
    if not os.path.exists(custom_gray_path):
        print(f"Không tìm thấy file {custom_gray_path}")
        return

    if not os.path.exists(opencv_gray_path):
        print(f"Không tìm thấy file {opencv_gray_path}")
        return

    # Đọc các ảnh
    custom_gray = cv2.imread(custom_gray_path, cv2.IMREAD_GRAYSCALE)
    opencv_gray = cv2.imread(opencv_gray_path, cv2.IMREAD_GRAYSCALE)

    # Đảm bảo cả hai ảnh có cùng kích thước
    if custom_gray.shape != opencv_gray.shape:
        print("Kích thước ảnh không khớp. Đang resize...")
        opencv_gray = cv2.resize(opencv_gray, (custom_gray.shape[1], custom_gray.shape[0]))

    # Tính các chỉ số so sánh
    metrics = calculate_metrics(custom_gray, opencv_gray)

    print("\nSo sánh kết quả chuyển đổi ảnh xám từ file:")
    print(f"  SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB")
    print(f"  MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}")

    # Tạo ảnh so sánh
    fig = create_comparison_image(
        custom_gray, opencv_gray,
        "Ảnh xám (tự cài đặt)",
        "Ảnh xám (OpenCV)",
        metrics
    )

    # Lưu ảnh so sánh
    comparison_path = os.path.join(comparison_dir, "compare_grayscale_from_files.jpg")
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)

    # Tạo ảnh hiển thị sự khác biệt với colormap khác
    diff = cv2.absdiff(custom_gray, opencv_gray)
    # Tăng độ tương phản của ảnh sai biệt để dễ nhìn
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=10)

    plt.figure(figsize=(10, 8))
    plt.imshow(diff_enhanced, cmap='jet')
    plt.colorbar(label='Độ khác biệt')
    plt.title(f'Sự khác biệt giữa hai phương pháp chuyển đổi ảnh xám\nMSE: {metrics["mse"]:.2f}, MAE: {metrics["mae"]:.2f}')
    plt.axis('off')
    plt.tight_layout()

    # Lưu ảnh sự khác biệt
    diff_path = os.path.join(comparison_dir, "grayscale_difference_heatmap.jpg")
    plt.savefig(diff_path, dpi=150)
    plt.close()

    print(f"\nĐã lưu ảnh so sánh tại: {comparison_path}")
    print(f"Đã lưu ảnh sự khác biệt tại: {diff_path}")

    # Tạo histogram để so sánh phân phối mức xám
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(custom_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Histogram ảnh xám (tự cài đặt)')
    plt.xlabel('Mức xám')
    plt.ylabel('Số lượng pixel')

    plt.subplot(1, 2, 2)
    plt.hist(opencv_gray.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('Histogram ảnh xám (OpenCV)')
    plt.xlabel('Mức xám')
    plt.ylabel('Số lượng pixel')

    plt.tight_layout()

    # Lưu histogram
    hist_path = os.path.join(comparison_dir, "grayscale_histograms.jpg")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print(f"Đã lưu histogram so sánh tại: {hist_path}")

    return metrics

def compare_grayscale_direct(img_path, comparison_dir='comparison'):
    """So sánh trực tiếp hai phương pháp chuyển đổi ảnh xám

    Args:
        img_path: Đường dẫn đến ảnh gốc
        comparison_dir: Thư mục lưu kết quả so sánh
    """
    # Đảm bảo thư mục comparison tồn tại
    ensure_directory_exists(comparison_dir)

    # Đọc ảnh gốc
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh từ {img_path}")
        return None

    print(f"Đã đọc ảnh {img_path} thành công. Kích thước: {img.shape}")

    # Đo thời gian thực thi của phương pháp tự cài đặt
    start_time = time.time()
    custom_gray = bgr_to_gray(img)
    custom_time = time.time() - start_time
    print(f"Thời gian chuyển đổi ảnh xám (tự cài đặt): {custom_time:.6f} giây")

    # Đo thời gian thực thi của OpenCV
    start_time = time.time()
    opencv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opencv_time = time.time() - start_time
    print(f"Thời gian chuyển đổi ảnh xám (OpenCV): {opencv_time:.6f} giây")

    # Tính tỷ lệ thời gian
    time_ratio = custom_time / opencv_time
    print(f"Tỷ lệ thời gian (tự cài đặt / OpenCV): {time_ratio:.2f}x")

    # Tính các chỉ số so sánh
    metrics = calculate_metrics(custom_gray, opencv_gray)

    print("\nSo sánh kết quả chuyển đổi ảnh xám (trực tiếp):")
    print(f"  SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB")
    print(f"  MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}")

    # Lưu các ảnh kết quả
    cv2.imwrite(os.path.join(comparison_dir, "direct_custom_gray.jpg"), custom_gray)
    cv2.imwrite(os.path.join(comparison_dir, "direct_opencv_gray.jpg"), opencv_gray)

    # Tạo ảnh hiển thị sự khác biệt
    diff = cv2.absdiff(custom_gray, opencv_gray)
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=10)  # Tăng độ tương phản để dễ nhìn
    cv2.imwrite(os.path.join(comparison_dir, "direct_difference.jpg"), diff_enhanced)

    # Tạo ảnh so sánh với matplotlib
    plt.figure(figsize=(15, 10))

    # Hiển thị ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Ảnh gốc')
    plt.axis('off')

    # Hiển thị ảnh xám (tự cài đặt)
    plt.subplot(2, 2, 2)
    plt.imshow(custom_gray, cmap='gray')
    plt.title(f'Ảnh xám (tự cài đặt)\nThời gian: {custom_time:.6f}s')
    plt.axis('off')

    # Hiển thị ảnh xám (OpenCV)
    plt.subplot(2, 2, 3)
    plt.imshow(opencv_gray, cmap='gray')
    plt.title(f'Ảnh xám (OpenCV)\nThời gian: {opencv_time:.6f}s')
    plt.axis('off')

    # Hiển thị sự khác biệt
    plt.subplot(2, 2, 4)
    plt.imshow(diff_enhanced, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Sự khác biệt (x10)\nSSIM: {metrics["ssim"]:.4f}, PSNR: {metrics["psnr"]:.2f}dB\nMSE: {metrics["mse"]:.2f}, MAE: {metrics["mae"]:.2f}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "direct_comparison_overview.jpg"), dpi=150)
    plt.close()

    # Tạo histogram để so sánh phân phối mức xám
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(custom_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Histogram ảnh xám (tự cài đặt)')
    plt.xlabel('Mức xám')
    plt.ylabel('Số lượng pixel')

    plt.subplot(1, 2, 2)
    plt.hist(opencv_gray.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('Histogram ảnh xám (OpenCV)')
    plt.xlabel('Mức xám')
    plt.ylabel('Số lượng pixel')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "direct_histograms.jpg"), dpi=150)
    plt.close()

    # Tạo file báo cáo
    with open(os.path.join(comparison_dir, "grayscale_report.txt"), 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO SO SÁNH CHUYỂN ĐỔI ẢNH XÁM\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Ảnh gốc: {img_path}\n")
        f.write(f"Kích thước: {img.shape} (sử dụng kích thước gốc, không resize)\n\n")

        f.write("THỜI GIAN THỰC THI\n")
        f.write("-" * 30 + "\n")
        f.write(f"Phương pháp tự cài đặt: {custom_time:.6f} giây\n")
        f.write(f"OpenCV: {opencv_time:.6f} giây\n")
        f.write(f"Tỷ lệ (tự cài đặt / OpenCV): {time_ratio:.2f}x\n\n")

        f.write("CHỈ SỐ CHẤT LƯỢNG\n")
        f.write("-" * 30 + "\n")
        f.write(f"SSIM: {metrics['ssim']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n\n")

        f.write("NHẬN XÉT\n")
        f.write("-" * 30 + "\n")
        if metrics['ssim'] > 0.98:
            f.write("Hai phương pháp cho kết quả rất tương đồng (SSIM > 0.98).\n")
        elif metrics['ssim'] > 0.95:
            f.write("Hai phương pháp cho kết quả khá tương đồng (SSIM > 0.95).\n")
        else:
            f.write("Có sự khác biệt đáng kể giữa hai phương pháp (SSIM < 0.95).\n")

        if time_ratio > 10:
            f.write("Phương pháp tự cài đặt chậm hơn đáng kể so với OpenCV.\n")
        elif time_ratio > 2:
            f.write("Phương pháp tự cài đặt chậm hơn so với OpenCV.\n")
        else:
            f.write("Phương pháp tự cài đặt có tốc độ tương đương với OpenCV.\n")

    print(f"\nĐã lưu kết quả so sánh trực tiếp vào thư mục {comparison_dir}")
    return metrics

def main():
    """Hàm chính"""
    print("SO SÁNH KẾT QUẢ GIỮA PHƯƠNG PHÁP TỰ CÀI ĐẶT VÀ OPENCV")
    print("=" * 60)

    # Thư mục chứa ảnh
    custom_dir = 'output'
    opencv_dir = 'output_opencv'
    comparison_dir = 'comparison'

    # Đảm bảo thư mục comparison tồn tại
    ensure_directory_exists(comparison_dir)

    print("\n1. SO SÁNH TRỰC TIẾP CHUYỂN ĐỔI ẢNH XÁM")
    print("-" * 40)
    # So sánh trực tiếp kết quả chuyển đổi ảnh xám
    img_path = 'img_minhhoa.jpg'
    grayscale_metrics = compare_grayscale_direct(img_path, comparison_dir)

    print("\n2. SO SÁNH TỪ CÁC FILE ĐÃ LƯU")
    print("-" * 40)
    # So sánh từ các file đã lưu
    compare_grayscale_from_files()

    # Đọc ảnh gốc để sử dụng trong so sánh ba chiều
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Không thể đọc ảnh gốc từ {img_path}")
        original_img = None
    else:
        print(f"Đã đọc ảnh gốc thành công. Kích thước: {original_img.shape}")

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

        # Tạo ảnh so sánh cơ bản
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

        # Tạo ảnh so sánh chi tiết (giống như so sánh ảnh xám)
        create_detailed_comparison(custom_img, opencv_img, filename, metrics, comparison_dir)

        # Tạo ảnh so sánh ba chiều (ảnh gốc, ảnh tự làm, ảnh opencv)
        if original_img is not None:
            # Chỉ tạo so sánh ba chiều nếu có ảnh gốc
            create_three_way_comparison(original_img, custom_img, opencv_img, filename, comparison_dir)
            print(f"  Đã tạo so sánh ba chiều: three_way_{filename}")

        print(f"  SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB")
        print(f"  MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}")

    # Tạo bảng tổng hợp
    print("\n3. BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH CÁC PHƯƠNG PHÁP XỬ LÝ ẢNH")
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
        f.write("BÁO CÁO TỔNG HỢP SO SÁNH PHƯƠNG PHÁP TỰ CÀI ĐẶT VÀ OPENCV\n")
        f.write("=" * 80 + "\n\n")

        # Thêm thông tin về so sánh chuyển đổi ảnh xám
        if grayscale_metrics:
            f.write("1. SO SÁNH CHUYỂN ĐỔI ẢNH XÁM\n")
            f.write("-" * 40 + "\n")
            f.write(f"SSIM: {grayscale_metrics['ssim']:.6f}\n")
            f.write(f"PSNR: {grayscale_metrics['psnr']:.2f} dB\n")
            f.write(f"MSE: {grayscale_metrics['mse']:.6f}\n")
            f.write(f"MAE: {grayscale_metrics['mae']:.6f}\n\n")

        f.write("2. BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH CÁC PHƯƠNG PHÁP XỬ LÝ ẢNH\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Tên file':<25} {'SSIM':<10} {'PSNR (dB)':<12} {'MSE':<10} {'MAE':<10}\n")
        f.write("-" * 80 + "\n")

        for item in comparison_table:
            f.write(f"{item['filename']:<25} {item['ssim']:<10.4f} {item['psnr']:<12.2f} {item['mse']:<10.2f} {item['mae']:<10.2f}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Trung bình':<25} {avg_ssim:<10.4f} {avg_psnr:<12.2f} {avg_mse:<10.2f} {avg_mae:<10.2f}\n")

    print("\n4. KẾT LUẬN")
    print("-" * 40)

    # Đưa ra kết luận về chuyển đổi ảnh xám
    if grayscale_metrics:
        print("Về chuyển đổi ảnh xám:")
        if grayscale_metrics['ssim'] > 0.98:
            print("- Hai phương pháp cho kết quả rất tương đồng (SSIM > 0.98)")
        elif grayscale_metrics['ssim'] > 0.95:
            print("- Hai phương pháp cho kết quả khá tương đồng (SSIM > 0.95)")
        else:
            print("- Có sự khác biệt đáng kể giữa hai phương pháp (SSIM < 0.95)")

    # Đưa ra kết luận chung
    print("\nVề tổng thể các phương pháp xử lý ảnh:")
    if avg_ssim > 0.95:
        print("- Nhìn chung, các phương pháp tự cài đặt cho kết quả tương đồng với OpenCV")
    else:
        print("- Có sự khác biệt giữa các phương pháp tự cài đặt và OpenCV")

    print(f"\nKết quả so sánh đã được lưu vào thư mục {comparison_dir}")
    print(f"Các ảnh so sánh đã được lưu với tiền tố 'compare_'")
    print(f"Báo cáo chi tiết về chuyển đổi ảnh xám: 'grayscale_report.txt'")
    print(f"Bảng tổng hợp tất cả các phương pháp: 'comparison_results.txt'")

if __name__ == "__main__":
    main()
