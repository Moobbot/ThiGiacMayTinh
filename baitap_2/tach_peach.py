import cv2
import numpy as np

# Hàm tăng kênh đỏ
def increase_red_channel(img, value):
    result = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = img[i, j]
            r = min(255, r + value)
            result[i, j] = (b, g, r)
    return result

# Hàm thêm nhãn văn bản lên ảnh
def add_label(img, text):
    labeled = img.copy()
    cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 255), 2, cv2.LINE_AA)
    return labeled

# ==== MAIN ====
# 1. Đọc ảnh
img = cv2.imread("images/peach.jpg")

# 2. Tăng kênh đỏ
red_boosted = increase_red_channel(img, 15)

# 3. Chuyển ảnh sang HSV
hsv = cv2.cvtColor(red_boosted, cv2.COLOR_BGR2HSV)

# 4. Định nghĩa vùng màu đỏ trong HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# 5. Tạo mask
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 6. Áp dụng mask để tách vật thể đỏ
red_object = cv2.bitwise_and(red_boosted, red_boosted, mask=mask)

# 7. Chuẩn hóa kích thước
height, width = img.shape[:2]
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 8. Thêm nhãn
img_label = add_label(img, "Original Image")
boosted_label = add_label(red_boosted, "Red Channel Boosted")
mask_label = add_label(mask_color, "Red Mask")
object_label = add_label(red_object, "Red Object Extracted")

# 9. Gộp ảnh
top_row = cv2.hconcat([img_label, boosted_label])
bottom_row = cv2.hconcat([mask_label, object_label])
final = cv2.vconcat([top_row, bottom_row])

# 10. Hiển thị kết quả
cv2.imshow('Combined Image with Labels', final)
cv2.waitKey(0)
cv2.destroyAllWindows()