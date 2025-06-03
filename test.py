import os
import cv2
import numpy as np
import csv

# ==== CẤU HÌNH ====
video_path = os.path.join("baitap_nc", "13512739_1080_1920_30fps.mp4")
log_file = 'orange_count_log.csv'
output_video_path = 'orange_detection_output.mp4'  # Đường dẫn video đầu ra
scale = 0.5
blur_ksize = (7, 7)
MIN_AREA = 400
DIST_THRESH = 40
MAX_MISSED = 5

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

# Resize frame đầu tiên
frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
frame_h, frame_w = frame.shape[:2]

# ==== KHỞI TẠO VIDEO WRITER ====
# Lấy FPS từ video gốc
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # FPS mặc định nếu không đọc được

# Khởi tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))

# ==== KHAI BÁO VÙNG ROI POLYGON (đã scale) ====
roi_polygon = np.array([
    [0, 400],   # top-left
    [540, 500],   # top-right
    [540, 640],   # bottom-right
    [0, 540]    # bottom-left
], dtype=np.int32)

# ==== TRACKER ====
next_id = 0
objects = {}
total_count = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ==== VÒNG LẶP ====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(frame, blur_ksize, 0)

    # Trừ nền + màu
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # === GÁN CENTROID VÀO OBJECT ===
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

                # Đếm nếu chưa đếm và nằm trong polygon
                if not data['counted']:
                    inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0
                    if inside:
                        total_count += 1
                        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        csv_writer.writerow([f"{time_sec:.2f}", total_count])
                        objects[obj_id]['counted'] = True
                        objects[obj_id]['id'] = total_count  # Store the count ID
                matched = True
                break
        if not matched:
            objects[next_id] = {'centroid': (cx, cy), 'counted': False, 'missed': 0}
            next_id += 1

    # Tăng missed và loại object nếu quá lâu không thấy
    for obj_id in list(objects.keys()):
        if obj_id not in updated_ids:
            objects[obj_id]['missed'] += 1
            if objects[obj_id]['missed'] > MAX_MISSED:
                del objects[obj_id]

    # Vẽ ROI polygon
    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 255), thickness=2)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_polygon], (255, 0, 255))
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Thông tin
    cv2.putText(frame, f"Count: {total_count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for obj_id, data in objects.items():
        if data.get('counted', False):
            cx, cy = data['centroid']
            count_id = data.get('id', 0)
            cv2.putText(frame, f"#{count_id}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    # Lưu frame vào video đầu ra
    out.write(frame)

    cv2.imshow("Detected Oranges", frame)
    cv2.imshow("Combined Mask", combined_mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# ==== CLEANUP ====
cap.release()
out.release()  # Đóng video writer
csv_file.close()
cv2.destroyAllWindows()

print(f"Video đã được lưu tại: {output_video_path}")
print(f"Tổng số cam đã đếm: {total_count}")
