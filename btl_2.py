"""
Orange Detection and Counting System

This module implements a real-time orange detection and counting system using computer vision techniques.
It combines traditional computer vision methods (color filtering, background subtraction) with deep learning
(Roboflow API) for robust orange detection in video streams.

Key Features:
- Real-time orange detection using both traditional CV and deep learning
- Background subtraction for motion detection
- Color-based filtering for orange detection
- Centroid tracking for object persistence
- Region of Interest (ROI) based counting
- CSV logging of detection counts
- Video output with visualization

Dependencies:
- OpenCV (cv2)
- NumPy
- tqdm
- inference_sdk
- PIL (Python Imaging Library)
- requests

Usage:
    python btl_2.py

The script processes a video file, detects oranges, tracks them across frames,
and counts them when they enter a defined ROI. Results are saved to a CSV file
and an annotated video output.
"""

import cv2
import numpy as np
import csv
import os
from tqdm import tqdm
from inference_sdk import InferenceHTTPClient
import tempfile
from PIL import Image
import time
import requests
from requests.exceptions import RequestException

# Initialize Roboflow client once for the entire session
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com", api_key="api_key"
)

# Constants for API retry mechanism
MAX_RETRIES = 3  # Maximum number of retry attempts for API calls
RETRY_DELAY = 1  # Delay between retries in seconds

def call_roboflow_api(img_path, max_retries=MAX_RETRIES):
    """
    Call Roboflow API with retry mechanism for orange detection.
    
    Args:
        img_path (str): Path to the image file to be processed
        max_retries (int): Maximum number of retry attempts (default: 3)
    
    Returns:
        dict or None: API response containing detection results, or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = CLIENT.infer(img_path, model_id="cv-0aa8x/1")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                return None

def fallback_detection(frame, combined_mask, MIN_AREA):
    """
    Fallback detection method using contour detection when API fails.
    
    Args:
        frame (numpy.ndarray): Input frame to process
        combined_mask (numpy.ndarray): Binary mask combining motion and color detection
        MIN_AREA (int): Minimum contour area threshold for detection
    
    Returns:
        tuple: (processed_frame, input_centroids, detection_count)
            - processed_frame: Frame with detection visualizations
            - input_centroids: List of detected orange centroids
            - detection_count: Number of oranges detected
    """
    input_centroids = []
    detection_count = 0
    
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                input_centroids.append((cx, cy))
                detection_count += 1
                
                # Draw bounding box and centroid
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
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

def validate_inputs(video_path, log_file, output_video):
    """
    Validate input files and paths, create necessary directories.
    
    Args:
        video_path (str): Path to input video file
        log_file (str): Path to output log file
        output_video (str): Path to output video file
    
    Returns:
        tuple: (video_path, log_file, output_video) with absolute paths
    
    Raises:
        FileNotFoundError: If video file doesn't exist
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
    Process a single frame to detect oranges using traditional CV methods.
    
    Args:
        frame (numpy.ndarray): Input frame
        scale (float): Scaling factor for frame resizing
        blur_ksize (tuple): Kernel size for Gaussian blur
        lower_orange (numpy.ndarray): Lower HSV bounds for orange color
        upper_orange (numpy.ndarray): Upper HSV bounds for orange color
        bg_subtractor: Background subtractor object
    
    Returns:
        tuple: (processed_frame, combined_mask, fg_mask)
            - processed_frame: Resized frame
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
    Detect oranges in frame using Roboflow API with fallback to traditional CV.
    
    Args:
        combined_mask (numpy.ndarray): Binary mask from traditional CV methods
        frame (numpy.ndarray): Input frame
        MIN_AREA (int): Minimum contour area threshold for fallback detection
    
    Returns:
        tuple: (processed_frame, input_centroids, detection_count)
            - processed_frame: Frame with detection visualizations
            - input_centroids: List of detected orange centroids
            - detection_count: Number of oranges detected
    """
    input_centroids = []
    detection_count = 0

    # Lưu frame tạm thời làm ảnh JPG
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_path = tmp.name
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(img_path)

    try:
        result = call_roboflow_api(img_path)
        if result is None:
            print("Using fallback detection method...")
            return fallback_detection(frame, combined_mask, MIN_AREA)
            
        for pred in result["predictions"]:
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            cx = x + w // 2
            cy = y + h // 2

            input_centroids.append((cx, cy))
            detection_count += 1

            # Vẽ bounding box và centroid
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(detection_count),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    except Exception as e:
        print(f"Error in detection: {e}")
        return fallback_detection(frame, combined_mask, MIN_AREA)

    return frame, input_centroids, detection_count


def main():
    """
    Main function to run the orange detection and counting system.
    """
    # ==== Configuration Parameters ====
    video_path = os.path.join("baitap_nc", "13512739_1080_1920_30fps.mp4")
    log_file = "orange_count_log_yolo_11.csv"
    output_video = "orange_detection_output_yolo_11.mp4"

    # Image processing parameters
    scale = 0.5  # Frame scaling factor
    blur_ksize = (7, 7)  # Gaussian blur kernel size
    MIN_AREA = 400  # Minimum contour area for detection
    DIST_THRESH = 40  # Distance threshold for centroid tracking
    MAX_MISSED = 5  # Maximum frames an object can be missed before removal
    playback_speed = 0.5  # Video playback speed multiplier

    # HSV color range for orange detection
    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])

    # Region of Interest (ROI) polygon coordinates (scaled)
    roi_polygon = np.array(
        [
            [0, 320],    # top-left corner
            [540, 480],  # top-right corner
            [540, 640],  # bottom-right corner
            [0, 540],    # bottom-left corner
        ],
        dtype=np.int32,
    )

    try:
        # Validate and prepare input/output paths
        video_path, log_file, output_video = validate_inputs(
            video_path, log_file, output_video
        )
        print(f"Using video: {video_path}")
        print(f"Log file: {log_file}")
        print(f"Output video: {output_video}")

        # ==== Initialize Video Processing ====
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Initialize background subtractor with optimized parameters
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

        # Initialize CSV logging
        with open(log_file, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Time(s)", "Count"])

            # Read first frame to get video properties
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read video frame.")

            print(f"Original video dimensions: {frame.shape[1]}x{frame.shape[0]} pixels")
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            frame_h, frame_w = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ==== Initialize Video Writer ====
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS if not available

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

            # ==== Initialize Tracking Variables ====
            next_id = 0
            objects = {}
            total_count = 0

            # Get total frames for progress tracking
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # ==== Main Processing Loop ====
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

                    # === Update Object Tracking ===
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

                                # Count object if it enters ROI and hasn't been counted
                                if not data["counted"]:
                                    inside = (
                                        cv2.pointPolygonTest(roi_polygon, (cx, cy), False)
                                        >= 0
                                    )
                                    if inside:
                                        total_count += 1
                                        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                                        csv_writer.writerow(
                                            [f"{time_sec:.2f}", total_count]
                                        )
                                        objects[obj_id]["counted"] = True
                                        objects[obj_id]["id"] = total_count
                                matched = True
                                break
                        if not matched:
                            objects[next_id] = {
                                "centroid": (cx, cy),
                                "counted": False,
                                "missed": 0,
                            }
                            next_id += 1

                    # Remove objects that haven't been seen for too long
                    for obj_id in list(objects.keys()):
                        if obj_id not in updated_ids:
                            objects[obj_id]["missed"] += 1
                            if objects[obj_id]["missed"] > MAX_MISSED:
                                del objects[obj_id]

                    # === Visualization ===
                    # Draw ROI polygon with semi-transparent overlay
                    cv2.polylines(
                        frame,
                        [roi_polygon],
                        isClosed=True,
                        color=(255, 0, 255),
                        thickness=2,
                    )
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [roi_polygon], (255, 0, 255))
                    alpha = 0.2
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Display count and object IDs
                    cv2.putText(
                        frame,
                        f"Count: {total_count}",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    for obj_id, data in objects.items():
                        if data.get("counted", False):
                            cx, cy = data["centroid"]
                            count_id = data.get("id", 0)
                            cv2.putText(
                                frame,
                                f"#{count_id}",
                                (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                2,
                            )

                    # Display frames
                    cv2.imshow("Detected Oranges", frame)
                    cv2.imshow("Combined Mask", combined_mask)
                    cv2.imshow("Foreground Mask", fg_mask)

                    # Write frame to output video
                    out.write(frame)
                    pbar.update(1)

                    # Handle frame timing and exit
                    wait_time = int(30 * (1.0 / playback_speed))
                    if cv2.waitKey(wait_time) & 0xFF == 27:
                        break

        # ==== Cleanup ====
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Output video saved to: {output_video}")
        print(f"Total oranges counted: {total_count}")
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
