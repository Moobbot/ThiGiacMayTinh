import cv2
import numpy as np

# Bước 1: Đọc ảnh
image = cv2.imread("images/peach.jpg")
blur = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow("1. Original Image", image)
cv2.imshow("2. Blurred Image", blur)

# Bước 2: Chuyển sang không gian màu HSV
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
cv2.imshow("3. HSV Image", hsv)

# Bước 3: Tạo mask cho vùng màu của quả đào
lower_peach1 = np.array([0, 100, 100])
upper_peach1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_peach1, upper_peach1)

lower_peach2 = np.array([170, 100, 100])
upper_peach2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_peach2, upper_peach2)

mask = cv2.bitwise_or(mask1, mask2)
cv2.imshow("4. Color Mask", mask)

# Bước 4: Làm sạch mask
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imshow("5. After Opening", mask_cleaned)

mask_dilated = cv2.dilate(mask_cleaned, kernel, iterations=2)
cv2.imshow("6. After Dilation", mask_dilated)

# Bước 5: Tìm contour và vẽ kết quả
result_image = image.copy()
contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 1000 < area < 10000:  # Giới hạn kích thước
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

# Bước 6: Hiển thị kết quả cuối cùng
cv2.putText(result_image, f"Peach count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imshow("7. Final Result", result_image)

# Đợi phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()