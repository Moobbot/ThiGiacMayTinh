import cv2
import numpy as np


# ==== Hàm tăng cường kênh đỏ ====
def increase_red_channel(img, value):
    result = img.copy()
    b, g, r = cv2.split(result)
    r = np.clip(r + value, 0, 255)
    result = cv2.merge([b, g, r])
    return result


# ==== Tính IOU giữa bounding box và hình tròn ====
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


# ==== Kiểm tra trùng hình tròn bằng IOU (không xét tâm) ====
def is_duplicate_circle(cx, cy, r, boxes, iou_thresh=0.3):
    for x1, y1, x2, y2 in boxes:
        if compute_iou((x1, y1, x2, y2), cx, cy, r) > iou_thresh:
            return True
    return False


def create_display_image(images, titles):
    n = len(images)
    if n == 0:
        return None

    # Tính số hàng và cột cần thiết
    nrows = (n - 1) // 3 + 1
    ncols = min(n, 3)

    # Chuẩn hóa kích thước ảnh
    height = 200
    width = 300

    # Tạo ảnh tổng hợp
    display = np.zeros((height * nrows, width * ncols, 3), dtype=np.uint8)

    for idx, (img, title) in enumerate(zip(images, titles)):
        i, j = idx // 3, idx % 3

        # Chuyển ảnh sang BGR nếu là ảnh grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize ảnh
        img = cv2.resize(img, (width, height))

        # Chèn ảnh vào vị trí tương ứng
        display[i * height : (i + 1) * height, j * width : (j + 1) * width] = img

        # Thêm tiêu đề
        cv2.putText(
            display,
            title,
            (j * width + 10, i * height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return display


# ==== MAIN ====
# 1. Load và tiền xử lý ảnh
img = cv2.imread("images/peach.jpg")
resized = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
red_boosted = increase_red_channel(resized, 15)

# 2. Chuyển HSV và tạo mask đỏ
hsv = cv2.cvtColor(red_boosted, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
red_mask = cv2.bitwise_or(mask1, mask2)

# 3. Morphology để làm sạch
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)

# 4. Chuẩn bị kết quả cuối cùng
result_img = resized.copy()

# ==== Blob Detection ====
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
peach_count = 0
detected_boxes = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    aspect_ratio = w / float(h)
    if 800 < area < 15000 and 0.5 < aspect_ratio < 1.8:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result_img,
            f"#{peach_count+1} Blob",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        peach_count += 1
        detected_boxes.append((x, y, x + w, y + h))

# ==== Hough Circles Detection ====
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
        if not is_duplicate_circle(x, y, r, detected_boxes):
            cv2.circle(result_img, (x, y), r, (255, 0, 0), 2)
            cv2.putText(
                result_img,
                f"#{peach_count+1} Circle",
                (x - r, y - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            peach_count += 1

# ==== Hiển thị tổng  số quả ====
cv2.putText(
    resized,
    f"Peach count: {peach_count}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (0, 0, 255),
    3,
)

# Tạo list ảnh và tiêu đề để hiển thị
display_images = [
    resized,  # Ảnh gốc
    red_boosted,  # Ảnh tăng cường đỏ
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),  # Ảnh HSV
    cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR),  # Mask đỏ
    cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR),  # Mask sau khi làm sạch
    result_img,  # Kết quả cuối cùng
]

display_titles = [
    "1. Original",
    "2. Red Boosted",
    "3. HSV",
    "4. Red Mask",
    "5. Cleaned Mask",
    "6. Final Result",
]

# Tạo và hiển thị ảnh tổng hợp
display = create_display_image(display_images, display_titles)
cv2.imshow("Peach Detection Steps", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
