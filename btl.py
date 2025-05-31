import cv2
import numpy as np

# 1. Khởi tạo bộ trừ nền (MOG2) và thiết lập ngưỡng màu cam HSV
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
lower_orange = np.array([10, 100, 20], dtype=np.uint8)   # HSV thấp nhất (H=10, S=100, V=20)
upper_orange = np.array([25, 255, 255], dtype=np.uint8)  # HSV cao nhất (H=25, S=255, V=255)

cap = cv2.VideoCapture('./baitap_nc/13512739_1080_1920_30fps.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Trừ nền để lấy mask vùng chuyển động
    fg_mask = bg_subtractor.apply(frame)
    # Chuyển mask về nhị phân (loại bỏ bóng mờ nếu có)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 3. Lọc màu cam trên không gian HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 4. Kết hợp mask chuyển động với mask màu cam
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)

    # 5. Xử lý hình thái để loại bỏ nhiễu và làm mịn vùng phát hiện
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Tìm các contour trên mask kết hợp
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Bỏ qua contour quá nhỏ (nếu cần thiết)
        if cv2.contourArea(cnt) < 300:  
            continue
        # Vẽ khung bao quanh contour (quả cam) lên frame gốc
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Mask', combined_mask)
    cv2.imshow('Detections', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát vòng lặp
        break

cap.release()
cv2.destroyAllWindows()
