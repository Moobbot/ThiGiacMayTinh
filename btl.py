"""
Orange Detection and Counting System

This script implements a computer vision system for detecting and counting oranges in a video stream.
It uses a combination of background subtraction, color filtering, and contour detection to identify
oranges and track them across frames. The system counts oranges that pass through a defined region
of interest (ROI).

Key Features:
- Real-time orange detection using color and motion analysis
- Background subtraction for improved detection
- Object tracking with centroid-based tracking
- Region of Interest (ROI) based counting
- CSV logging of detection counts
- Visual output with bounding boxes and count labels
- Progress bar for processing status

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - tqdm
    - csv
    - os

Usage:
    Run the script directly. The video path, log file, and output video paths are configured
    in the script. Adjust parameters like scale, blur size, and color thresholds as needed.

Parameters:
    - scale: Scaling factor for input frames (default: 0.5)
    - blur_ksize: Kernel size for Gaussian blur (default: (7,7))
    - MIN_AREA: Minimum contour area for orange detection (default: 400)
    - DIST_THRESH: Distance threshold for object tracking (default: 40)
    - MAX_MISSED: Maximum frames an object can be missed before removal (default: 5)
    - playback_speed: Video playback speed multiplier (default: 0.5)
"""

import cv2
import numpy as np
import csv
import os
from tqdm import tqdm


def validate_inputs(video_path, log_file, output_video):
    """
    Validate input files and create necessary directories.
    
    Args:
        video_path (str): Path to the input video file
        log_file (str): Path to the output CSV log file
        output_video (str): Path to the output video file
        
    Returns:
        tuple: Absolute paths for video_path, log_file, and output_video
        
    Raises:
        FileNotFoundError: If the input video file doesn't exist
    """
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
    """
    Process a single frame to detect oranges using color and motion analysis.
    
    Args:
        frame (numpy.ndarray): Input frame from video
        scale (float): Scaling factor for the frame
        blur_ksize (tuple): Kernel size for Gaussian blur
        lower_orange (numpy.ndarray): Lower HSV bounds for orange color
        upper_orange (numpy.ndarray): Upper HSV bounds for orange color
        bg_subtractor: Background subtractor object
        
    Returns:
        tuple: (processed_frame, combined_mask, fg_mask)
            - processed_frame: Frame after scaling
            - combined_mask: Binary mask combining motion and color detection
            - fg_mask: Foreground mask from background subtraction
    """
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

    return frame, combined_mask, fg_mask


def detect_oranges(combined_mask, frame, MIN_AREA):
    """
    Detect oranges in the processed frame using contour detection.
    
    Args:
        combined_mask (numpy.ndarray): Binary mask from process_frame
        frame (numpy.ndarray): Original frame for drawing
        MIN_AREA (int): Minimum contour area to consider as an orange
        
    Returns:
        tuple: (frame, input_centroids, detection_count)
            - frame: Frame with detection boxes and labels
            - input_centroids: List of (x,y) coordinates for detected oranges
            - detection_count: Number of oranges detected in current frame
    """
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


def main():
    """
    Main function to run the orange detection and counting system.
    """
    # ==== THIẾT LẬP ====
    video_path = os.path.join("baitap_nc", "13512739_1080_1920_30fps.mp4")
    log_file = "orange_count_log.csv"
    output_video = "orange_detection_output.mp4"
    scale = 0.5
    blur_ksize = (7, 7)
    MIN_AREA = 400
    DIST_THRESH = 40
    MAX_MISSED = 5
    playback_speed = 0.5

    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])

    # ==== KHAI BÁO VÙNG ROI POLYGON (đã scale) ====
    roi_polygon = np.array([
        [0, 320],   # top-left
        [540, 480],   # top-right
        [540, 640],   # bottom-right
        [0, 540]    # bottom-left
    ], dtype=np.int32)

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
            history=500, varThreshold=50, detectShadows=False
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ==== KHỞI TẠO VIDEO WRITER ====
            # Lấy FPS từ video gốc
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # FPS mặc định nếu không đọc được

            # Khởi tạo VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho MP4
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

                    # Process frame and detect oranges
                    frame, combined_mask, fg_mask = process_frame(
                        frame, scale, blur_ksize, lower_orange, upper_orange, bg_subtractor
                    )
                    frame, input_centroids, detection_count = detect_oranges(
                        combined_mask, frame, MIN_AREA
                    )

                    # === GÁN CENTROID VÀO OBJECT ===
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

                                # Đếm nếu chưa đếm và nằm trong polygon
                                if not data["counted"]:
                                    inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0
                                    if inside:
                                        total_count += 1
                                        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                                        csv_writer.writerow([f"{time_sec:.2f}", total_count])
                                        objects[obj_id]["counted"] = True
                                        objects[obj_id]["id"] = total_count  # Store the count ID
                                matched = True
                                break
                        if not matched:
                            objects[next_id] = {
                                "centroid": (cx, cy),
                                "counted": False,
                                "missed": 0,
                            }
                            next_id += 1

                    # Tăng missed và loại object nếu quá lâu không thấy
                    for obj_id in list(objects.keys()):
                        if obj_id not in updated_ids:
                            objects[obj_id]["missed"] += 1
                            if objects[obj_id]["missed"] > MAX_MISSED:
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
                        if data.get("counted", False):
                            cx, cy = data["centroid"]
                            count_id = data.get("id", 0)
                            cv2.putText(frame, f"#{count_id}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # Display frames
                    cv2.imshow("Detected Oranges", frame)
                    cv2.imshow("Combined Mask", combined_mask)
                    cv2.imshow("Foreground Mask", fg_mask)  # Hiển thị video tách nền

                    out.write(frame)
                    pbar.update(1)

                    wait_time = int(30 * (1.0 / playback_speed))
                    if cv2.waitKey(wait_time) & 0xFF == 27:
                        break

        # ==== CLEANUP ====
        cap.release()
        out.release()  # Đóng video writer
        cv2.destroyAllWindows()

        print(f"Video đã được lưu tại: {output_video}")
        print(f"Tổng số cam đã đếm: {total_count}")
        print(f"Processing complete. Total oranges counted: {total_count}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if "cap" in locals():
            cap.release()
        if "out" in locals():
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
