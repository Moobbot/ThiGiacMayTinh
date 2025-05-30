import cv2
import numpy as np


# ==== Hàm tăng cường kênh đỏ ====
def increase_red_channel(img, value):
    result = img.copy()
    b, g, r = cv2.split(result)
    r = np.clip(r + value, 0, 255)
    result = cv2.merge([b, g, r])
    return result


# ==== Hàm tính IOU giữa bounding box và hình tròn ====
def compute_iou(boxA, circleX, circleY, r):
    boxB = [circleX - r, circleY - r, circleX + r, circleY + r]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


# ==== MAIN ====

# 1. Đọc ảnh
img = cv2.imread("images/peach.jpg")
resized = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
red_boosted = increase_red_channel(resized, 15)

# 2. Chuyển sang HSV
hsv = cv2.cvtColor(red_boosted, cv2.COLOR_BGR2HSV)

# 3. Mask đỏ trong HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# 4. Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)

# 5. Blob detection
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
peach_count = 0
detected_boxes = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    aspect_ratio = w / float(h)
    if 800 < area < 15000 and 0.5 < aspect_ratio < 1.8:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        peach_count += 1
        detected_boxes.append((x, y, x + w, y + h))

# 6. HoughCircles (giới hạn trong vùng đỏ)
gray = cv2.cvtColor(red_boosted, cv2.COLOR_BGR2GRAY)
masked_gray = cv2.bitwise_and(gray, gray, mask=cleaned_mask)

circles = cv2.HoughCircles(
    masked_gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=40,
    minRadius=20,
    maxRadius=60,
)

if circles is not None:
    circles = np.uint16(np.around(circles[0, :]))
    for x, y, r in circles:
        is_duplicate = False
        for x1, y1, x2, y2 in detected_boxes:
            iou = compute_iou((x1, y1, x2, y2), x, y, r)
            if iou > 0.3:  # Ngưỡng IOU để coi là trùng
                is_duplicate = True
                break
        if not is_duplicate:
            cv2.circle(resized, (x, y), r, (255, 0, 0), 2)
            peach_count += 1

# 7. Hiển thị kết quả
cv2.putText(
    resized,
    f"Peach count: {peach_count}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (0, 0, 255),
    3,
)

cv2.imshow("Peach Detection (IOU + Hybrid)", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
