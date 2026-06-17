"""
YOLO Object Detection with Simple Tracking
===========================================
This script performs real-time object detection on a video file using YOLOv3,
with simple centroid-based tracking to maintain object IDs across frames.

Files required:
- yolov3.weights (YOLO weights)
- yolov3.cfg (YOLO config)
- coco.names (class labels)
- test_video.mp4 (input video)

Output:
- output_video.mp4 (detected video with bounding boxes and tracking IDs)
"""

import cv2
import numpy as np
import time


# =============================================================================
# Configuration
# =============================================================================

# File paths
WEIGHTS_PATH = "yolov3.weights"
CONFIG_PATH = "yolov3.cfg"
CLASSES_PATH = "coco.names"
INPUT_VIDEO = "test_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider a detection
NMS_THRESHOLD = 0.4         # Non-maximum suppression threshold

# Tracking parameters
MAX_DISTANCE = 50            # Maximum distance to match detections between frames
# If a detection is more than MAX_DISTANCE pixels away from any previous detection,
# it gets assigned a new ID

# Output display
WINDOW_NAME = "YOLO Object Detection"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


# =============================================================================
# Helper Functions
# =============================================================================

def load_class_names(filepath):
    """
    Load COCO class names from the .names file.

    Args:
        filepath: Path to the coco.names file

    Returns:
        List of class names
    """
    with open(filepath, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def load_yolo_network(config_path, weights_path):
    """
    Load YOLO neural network from config and weights files.

    Args:
        config_path: Path to yolov3.cfg
        weights_path: Path to yolov3.weights

    Returns:
        YOLO network object, output layer names
    """
    # Load YOLO network using OpenCV's DNN module
    # Reads the Darknet model configuration and pre-trained weights
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Get output layer names (YOLO has 3 output layers for different scales)
    # Each layer outputs detections at a different resolution
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers


def get_output_boxes(outputs, img_width, img_height, confidence_threshold):
    """
    Process YOLO network outputs to extract bounding boxes.

    Args:
        outputs: Raw outputs from YOLO network
        img_width: Width of input image
        img_height: Height of input image
        confidence_threshold: Minimum confidence for detections

    Returns:
        boxes: List of bounding boxes [x, y, w, h]
        confidences: List of confidence scores
        class_ids: List of class IDs
    """
    boxes = []
    confidences = []
    class_ids = []

    # Each output layer contains detections for all anchor boxes
    # Detection format: [center_x, center_y, width, height, object_conf, class1, class2, ...]
    for output in outputs:
        for detection in output:
            # Extract class probabilities (all classes after the first 5 values)
            scores = detection[5:]

            # Find the class with highest probability
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter by confidence threshold
            if confidence > confidence_threshold:
                # Scale bounding box coordinates back to original image size
                # YOLO outputs normalized coordinates (0-1), so we multiply by image dimensions
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)

                # Convert from center format to top-left corner format
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def apply_nms(boxes, confidences, nms_threshold):
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.

    Non-Maximum Suppression keeps only the box with highest confidence
    when multiple boxes overlap significantly for the same object.

    Args:
        boxes: List of bounding boxes
        confidences: List of confidence scores
        nms_threshold: NMS threshold (IoU threshold for suppression)

    Returns:
        indices: Indices of boxes to keep after NMS
    """
    # Apply Non-Maximum Suppression using OpenCV's DNN module
    # This filters out overlapping boxes based on the NMS threshold
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, nms_threshold)
    return indices


def calculate_centroid(box):
    """
    Calculate the centroid (center point) of a bounding box.

    Args:
        box: Bounding box [x, y, w, h] where x,y is top-left corner

    Returns:
        (cx, cy): Centroid coordinates
    """
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Distance between the two points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def match_detections_to_tracks(detections, previous_tracks, max_distance):
    """
    Match current detections to existing tracks using Euclidean distance.

    This implements simple centroid-based tracking:
    - For each detection, find the closest previous track
    - If within MAX_DISTANCE, link them (same ID)
    - Otherwise, assign a new unique ID

    Args:
        detections: List of detected boxes [x, y, w, h]
        previous_tracks: Dictionary of {track_id: centroid}
        max_distance: Maximum distance for a valid match

    Returns:
        matches: Dictionary mapping detection_index to track_id
        new_track_ids: Set of newly assigned track IDs
    """
    matches = {}

    # If no previous tracks, assign new IDs starting from 0
    if not previous_tracks:
        new_id = 0
        for det_idx in range(len(detections)):
            matches[det_idx] = new_id
            new_id += 1
        return matches, set()

    # Get all currently used track IDs
    used_track_ids = set(previous_tracks.keys())

    # Match each detection to the closest previous track
    for det_idx, det_box in enumerate(detections):
        det_centroid = calculate_centroid(det_box)

        # Find closest previous track
        best_track_id = None
        best_distance = float('inf')

        for track_id, track_centroid in previous_tracks.items():
            distance = euclidean_distance(det_centroid, track_centroid)
            if distance < best_distance and distance <= max_distance:
                best_distance = distance
                best_track_id = track_id

        # If a match was found, assign that track ID
        if best_track_id is not None:
            matches[det_idx] = best_track_id
            used_track_ids.discard(best_track_id)

    # Assign new IDs to unmatched detections
    new_track_ids = set()
    if used_track_ids:
        # Reuse IDs from tracks that disappeared (lowest available)
        new_id = min(used_track_ids)
    else:
        # Start new IDs from after the highest existing ID
        new_id = max(previous_tracks.keys()) + 1 if previous_tracks else 0

    for det_idx in range(len(detections)):
        if det_idx not in matches:
            matches[det_idx] = new_id
            new_track_ids.add(new_id)
            new_id += 1

    return matches, new_track_ids


def draw_detections(img, boxes, confidences, class_ids, track_ids, class_names):
    """
    Draw bounding boxes, class labels, and tracking IDs on the image.

    Args:
        img: Input image (frame)
        boxes: List of bounding boxes
        confidences: List of confidence scores
        class_ids: List of class IDs
        track_ids: List of tracking IDs
        class_names: List of class names

    Returns:
        Image with drawn detections
    """
    # Generate consistent random colors for each class
    # Seed ensures same colors across all frames
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    for i, box in enumerate(boxes):
        x, y, w, h = box
        confidence = confidences[i]
        class_id = class_ids[i]
        track_id = track_ids[i]

        # Get unique color for this class
        color = tuple(map(int, colors[class_id]))

        # Draw rectangle around the detected object
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Create label string: "ID:X ClassName: Confidence"
        label = f"ID:{track_id} {class_names[class_id]}: {confidence:.2f}"

        # Get text dimensions for drawing background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )

        # Draw filled rectangle behind text for better readability
        label_y = max(y, label_height + 10)
        cv2.rectangle(
            img,
            (x, label_y - label_height - 4),
            (x + label_width, label_y + baseline - 4),
            color,
            -1
        )

        # Draw the label text in white
        cv2.putText(
            img,
            label,
            (x, label_y - 2),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS
        )

    return img


def draw_fps(img, fps, frame_count):
    """
    Draw FPS counter and frame count on the image.

    Args:
        img: Input image (frame)
        fps: Current frames per second
        frame_count: Total frames processed so far

    Returns:
        Image with FPS overlay
    """
    fps_text = f"FPS: {fps:.1f} | Frame: {frame_count}"

    # Draw dark background rectangle for visibility
    cv2.rectangle(img, (5, 5), (240, 35), (0, 0, 0), -1)

    # Draw FPS text in green
    cv2.putText(
        img,
        fps_text,
        (10, 25),
        FONT,
        0.6,
        (0, 255, 0),
        2
    )

    return img


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main function to run YOLO object detection with tracking on video.

    Steps:
    1. Load YOLO model and COCO class names
    2. Open input video and prepare output writer
    3. Process each frame:
       a. Prepare image blob for YOLO
       b. Run forward pass through network
       c. Extract and filter bounding boxes
       d. Apply Non-Maximum Suppression
       e. Match detections to existing tracks (centroid tracking)
       f. Draw bounding boxes and labels on frame
       g. Display frame and check for quit key
    4. Release resources and save output
    """
    print("=" * 60)
    print("YOLO Object Detection with Simple Tracking")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load YOLO model and class names
    # -------------------------------------------------------------------------
    print("\n[1] Loading YOLO model and class names...")

    # Load COCO class names from file
    class_names = load_class_names(CLASSES_PATH)
    print(f"    Loaded {len(class_names)} class names")

    # Load YOLO neural network from config and weights
    net, output_layers = load_yolo_network(CONFIG_PATH, WEIGHTS_PATH)
    print(f"    Loaded YOLO with {len(output_layers)} output layers")

    # Configure to use CPU (can be changed to DNN_BACKEND_CUDA for GPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # -------------------------------------------------------------------------
    # Step 2: Open input video and prepare output writer
    # -------------------------------------------------------------------------
    print("\n[2] Opening video files...")

    # Open input video file
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{INPUT_VIDEO}'")
        return

    # Get video properties (dimensions, FPS, frame count)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"    Input video: {INPUT_VIDEO}")
    print(f"    Resolution: {width}x{height}")
    print(f"    FPS: {fps}")
    print(f"    Total frames: {total_frames}")

    # Create VideoWriter object for saving output
    # Uses mp4v codec, matches input FPS and resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"    Output video: {OUTPUT_VIDEO}")

    # -------------------------------------------------------------------------
    # Step 3: Process video frames
    # -------------------------------------------------------------------------
    print("\n[3] Processing video frames...")

    # Tracking variables
    previous_tracks = {}  # {track_id: centroid} - stores last known position of each tracked object
    next_track_id = 0     # Counter for assigning new IDs
    frame_count = 0       # Total frames processed

    # FPS calculation
    start_time = time.time()
    fps_update_interval = 30  # Print progress every N frames

    # Create window for displaying output
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        # Read next frame from video
        ret, frame = cap.read()

        # If no frame returned, we've reached the end of video
        if not ret:
            print("    End of video reached")
            break

        frame_count += 1

        # Get current frame dimensions
        img_height, img_width = frame.shape[:2]

        # -------------------------------------------------------------------------
        # Step 3a: Prepare image for YOLO
        # -------------------------------------------------------------------------
        # Convert image to blob format required by YOLO
        # - Scale to 1/255 (normalize to 0-1 range)
        # - Resize to 416x416 (YOLO input size)
        # - Swap RB channels (BGR to RGB)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )

        # Set blob as input to the network
        net.setInput(blob)

        # -------------------------------------------------------------------------
        # Step 3b: Run forward pass through YOLO
        # -------------------------------------------------------------------------
        # Forward pass returns detections from all output layers
        outputs = net.forward(output_layers)

        # -------------------------------------------------------------------------
        # Step 3c: Extract bounding boxes from outputs
        # -------------------------------------------------------------------------
        boxes, confidences, class_ids = get_output_boxes(
            outputs, img_width, img_height, CONFIDENCE_THRESHOLD
        )

        # -------------------------------------------------------------------------
        # Step 3d: Apply Non-Maximum Suppression
        # -------------------------------------------------------------------------
        indices = apply_nms(boxes, confidences, NMS_THRESHOLD)

        # Filter to keep only boxes that passed NMS
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else [indices]
            boxes = [boxes[i] for i in indices]
            confidences = [confidences[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]
        else:
            boxes = []
            confidences = []
            class_ids = []

        # -------------------------------------------------------------------------
        # Step 3e: Simple centroid-based tracking
        # -------------------------------------------------------------------------
        if boxes:
            # Match current detections to previous tracks
            matches, new_track_ids = match_detections_to_tracks(
                boxes, previous_tracks, MAX_DISTANCE
            )

            # Update tracking information for next frame
            previous_tracks = {}
            track_ids = []

            for det_idx, box in enumerate(boxes):
                track_id = matches[det_idx]
                track_ids.append(track_id)
                previous_tracks[track_id] = calculate_centroid(box)

            # Update next available track ID
            next_track_id = max(track_ids) + 1 if track_ids else 0
        else:
            # No detections in this frame - clear tracks
            previous_tracks = {}
            track_ids = []

        # -------------------------------------------------------------------------
        # Step 3f: Draw detections on frame
        # -------------------------------------------------------------------------
        frame = draw_detections(
            frame, boxes, confidences, class_ids, track_ids, class_names
        )

        # Calculate current FPS
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Draw FPS counter
        frame = draw_fps(frame, current_fps, frame_count)

        # Write frame to output video file
        out.write(frame)

        # Display frame in window
        cv2.imshow(WINDOW_NAME, frame)

        # Print progress periodically
        if frame_count % fps_update_interval == 0:
            print(f"    Processed frame {frame_count}/{total_frames} | FPS: {current_fps:.1f}")

        # -------------------------------------------------------------------------
        # Step 3g: Check for quit key ('q')
        # -------------------------------------------------------------------------
        # waitKey(1) waits 1ms for key press, & 0xFF masks to get ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("    User pressed 'q' to quit")
            break

    # -------------------------------------------------------------------------
    # Step 4: Cleanup and save output
    # -------------------------------------------------------------------------
    print("\n[4] Cleaning up...")

    # Release video capture and writer resources
    cap.release()
    out.release()

    # Close the display window
    cv2.destroyAllWindows()

    # Print final summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"  Total frames processed: {frame_count}")
    print(f"  Average FPS: {current_fps:.1f}")
    print(f"  Output saved to: {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()