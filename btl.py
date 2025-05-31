import cv2
import numpy as np
import csv

# ==== THIẾT LẬP ====
video_path = './baitap_nc/13512739_1080_1920_30fps.mp4'
log_file = 'orange_count_log.csv'
output_video = 'output_video.mp4'  # Tên file video output
scale = 0.5
blur_ksize = (7, 7)
MIN_AREA = 400
DIST_THRESH = 40
MAX_MISSED = 5
playback_speed = 0.5  # Tốc độ phát (1.0 là bình thường, < 1.0 là chậm hơn)

lower_orange = np.array([10, 100, 20])
upper_orange = np.array([25, 255, 255])

# ==== KHỞI TẠO ====
cap = cv2.VideoCapture(video_path)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

csv_file = open(log_file, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time(s)', 'Count'])

ret, frame = cap.read()
if not ret:
    print("Không thể đọc video.")
    exit()
print(f"Kích thước video gốc: {frame.shape[1]}x{frame.shape[0]} pixels")
frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
frame_h, frame_w = frame.shape[:2]
roi_top = int(500 * scale)  # Nếu scale = 0.5 → 250
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Thiết lập video writer
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

# ==== CENTROID TRACKER ====
next_id = 0
objects = {}  # object_id: {'centroid': (x,y), 'counted': bool, 'missed': int}
total_count = 0
detection_count = 0  # Thêm biến đếm số thứ tự phát hiện

# ==== VÒNG LẶP ====
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(frame, blur_ksize, 0)

    # Trừ nền + lọc màu cam
    fg_mask = bg_subtractor.apply(blurred)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    input_centroids = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        input_centroids.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        # Thêm số thứ tự
        detection_count += 1
        cv2.putText(frame, str(detection_count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Cập nhật object tracker
    updated_ids = set()
    for cx, cy in input_centroids:
        matched = False
        for obj_id, data in objects.items():
            ox, oy = data['centroid']
            dist = np.hypot(cx - ox, cy - oy)
            if dist < DIST_THRESH:
                objects[obj_id]['centroid'] = (cx, cy)
                objects[obj_id]['missed'] = 0
                updated_ids.add(obj_id)

                # Đếm nếu chưa đếm và đi vào ROI
                if not data['counted'] and cy > roi_top:
                    total_count += 1
                    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    csv_writer.writerow([f"{time_sec:.2f}", total_count])
                    objects[obj_id]['counted'] = True
                matched = True
                break
        if not matched:
            # Tạo object mới
            objects[next_id] = {'centroid': (cx, cy), 'counted': False, 'missed': 0}
            next_id += 1

    # Tăng missed cho object không cập nhật
    for obj_id in list(objects.keys()):
        if obj_id not in updated_ids:
            objects[obj_id]['missed'] += 1
            if objects[obj_id]['missed'] > MAX_MISSED:
                del objects[obj_id]

    # Vẽ ROI
    cv2.line(frame, (0, roi_top), (frame_w, roi_top), (0, 255, 255), 2)
    cv2.putText(frame, f"Count: {total_count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Detected Oranges", frame)
    cv2.imshow("Combined Mask", combined_mask)
    
    # Lưu frame vào video
    out.write(frame)
    
    # Điều chỉnh thời gian chờ dựa trên tốc độ phát
    wait_time = int(30 * (1.0 / playback_speed))  # 30ms là thời gian chờ mặc định
    if cv2.waitKey(wait_time) & 0xFF == 27:
        break

# ==== KẾT THÚC ====
cap.release()
out.release()  # Đóng video writer
csv_file.close()
cv2.destroyAllWindows()
