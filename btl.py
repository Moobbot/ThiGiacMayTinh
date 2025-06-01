import cv2
import numpy as np
import csv
import os
from tqdm import tqdm


def validate_inputs(video_path, log_file, output_video):
    """Validate input files and paths."""
    # Convert to absolute path
    video_path = os.path.abspath(video_path)
    log_file = os.path.abspath(log_file)
    output_video = os.path.abspath(output_video)
    
    print(f"Checking video path: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create directories if they don't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
    
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return video_path, log_file, output_video


def process_frame(frame, scale, blur_ksize, lower_orange, upper_orange, bg_subtractor):
    """Process a single frame to detect oranges."""
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(frame, blur_ksize, 0)

    # Background subtraction and color filtering
    fg_mask = bg_subtractor.apply(blurred)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    return frame, combined_mask


def detect_oranges(combined_mask, frame, MIN_AREA):
    """Detect oranges in the processed frame."""
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    input_centroids = []
    detection_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        input_centroids.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        detection_count += 1
        cv2.putText(
            frame,
            str(detection_count),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame, input_centroids, detection_count


# ==== THIẾT LẬP ====
video_path = os.path.join("baitap_nc", "13512739_1080_1920_30fps.mp4")
log_file = "orange_count_log.csv"
output_video = "output_video.mp4"
scale = 0.5
blur_ksize = (7, 7)
MIN_AREA = 400
DIST_THRESH = 40
MAX_MISSED = 5
playback_speed = 0.5

lower_orange = np.array([10, 100, 20])
upper_orange = np.array([25, 255, 255])

try:
    # Validate inputs
    video_path, log_file, output_video = validate_inputs(video_path, log_file, output_video)
    print(f"Using video: {video_path}")
    print(f"Log file: {log_file}")
    print(f"Output video: {output_video}")

    # ==== KHỞI TẠO ====
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=600, varThreshold=50, detectShadows=False
    )

    with open(log_file, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Time(s)", "Count"])

        ret, frame = cap.read()
        if not ret:
            raise ValueError("Không thể đọc video.")

        print(f"Kích thước video gốc: {frame.shape[1]}x{frame.shape[0]} pixels")
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame_h, frame_w = frame.shape[:2]
        roi_top = int(600 * scale)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Thiết lập video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

        # ==== CENTROID TRACKER ====
        next_id = 0
        objects = {}
        total_count = 0

        # Get total frames for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ==== VÒNG LẶP ====
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame, combined_mask = process_frame(
                    frame, scale, blur_ksize, lower_orange, upper_orange, bg_subtractor
                )
                frame, input_centroids, detection_count = detect_oranges(
                    combined_mask, frame, MIN_AREA
                )

                # Update object tracker
                updated_ids = set()
                for cx, cy in input_centroids:
                    matched = False
                    for obj_id, data in objects.items():
                        ox, oy = data["centroid"]
                        dist = np.hypot(cx - ox, cy - oy)
                        if dist < DIST_THRESH:
                            objects[obj_id]["centroid"] = (cx, cy)
                            objects[obj_id]["missed"] = 0
                            updated_ids.add(obj_id)

                            if not data["counted"] and cy > roi_top:
                                total_count += 1
                                time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                                csv_writer.writerow([f"{time_sec:.2f}", total_count])
                                objects[obj_id]["counted"] = True
                            matched = True
                            break
                    if not matched:
                        objects[next_id] = {
                            "centroid": (cx, cy),
                            "counted": False,
                            "missed": 0,
                        }
                        next_id += 1

                # Update missed objects
                for obj_id in list(objects.keys()):
                    if obj_id not in updated_ids:
                        objects[obj_id]["missed"] += 1
                        if objects[obj_id]["missed"] > MAX_MISSED:
                            del objects[obj_id]

                # Draw ROI and count
                cv2.line(frame, (0, roi_top), (frame_w, roi_top), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"Count: {total_count}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Detected Oranges", frame)
                cv2.imshow("Combined Mask", combined_mask)

                out.write(frame)
                pbar.update(1)

                wait_time = int(30 * (1.0 / playback_speed))
                if cv2.waitKey(wait_time) & 0xFF == 27:
                    break

    # ==== KẾT THÚC ====
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Total oranges counted: {total_count}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    if "cap" in locals():
        cap.release()
    if "out" in locals():
        out.release()
    cv2.destroyAllWindows()
