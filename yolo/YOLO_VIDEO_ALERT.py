import cv2
import torch
import numpy as np
from pathlib import Path
import argparse

# Argument parser for file inputs
parser = argparse.ArgumentParser(description="Object detection and anomaly flagging")
parser.add_argument('--input1', type=str, required=True, help="Path to video or first image")
parser.add_argument('--input2', type=str, help="Path to second image (optional, for image mode)")
args = parser.parse_args()

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image):
    """Run object detection on an image and return detections."""
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max, confidence, class]
    return detections

def get_object_center(detections):
    """Calculate the center of detected objects."""
    centers = []
    for det in detections:
        x_min, y_min, x_max, y_max = det[:4]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        centers.append((center_x, center_y, int(det[5])))  # (x, y, class_id)
    return centers

def flag_misplaced_objects(ref_centers, curr_centers, threshold=50):
    """Flag objects that have moved significantly or are new as misplaced."""
    misplaced = []
    for curr in curr_centers:
        match = False
        for ref in ref_centers:
            if curr[2] == ref[2]:  # Same class
                dist = np.sqrt((curr[0] - ref[0])**2 + (curr[1] - ref[1])**2)
                if dist < threshold:
                    match = True
                    break
        if not match:
            misplaced.append(curr)
    return misplaced

def draw_boxes(image, detections, misplaced_centers):
    """Draw bounding boxes, highlighting misplaced objects in red."""
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2, int(cls))
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = (0, 0, 255) if center in misplaced_centers else (0, 255, 0)  # Red for misplaced, green for normal
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Process based on input type
input1_path = Path(args.input1)
input2_path = Path(args.input2) if args.input2 else None

if input1_path.suffix in ['.mp4', '.avi', '.mov']:  # Video mode
    cap = cv2.VideoCapture(str(input1_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get first frame as reference
    ret, ref_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        exit()
    ref_detections = detect_objects(ref_frame)
    ref_centers = get_object_center(ref_detections)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in current frame
        detections = detect_objects(frame)
        centers = get_object_center(detections)

        # Flag misplaced objects
        misplaced = flag_misplaced_objects(ref_centers, centers)

        # Draw boxes and save frame
        output_frame = draw_boxes(frame.copy(), detections, misplaced)
        out.write(output_frame)

        # Display (optional, comment out if not needed)
        cv2.imshow('Video', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete. Output saved as 'output_video.mp4'.")

else:  # Image mode
    img1 = cv2.imread(str(input1_path))
    if img1 is None:
        print("Error: Could not load first image.")
        exit()

    # Detect objects in first image (reference)
    ref_detections = detect_objects(img1)
    ref_centers = get_object_center(ref_detections)

    if input2_path:  # Two images provided
        img2 = cv2.imread(str(input2_path))
        if img2 is None:
            print("Error: Could not load second image.")
            exit()

        # Detect objects in second image
        detections = detect_objects(img2)
        centers = get_object_center(detections)

        # Flag misplaced objects
        misplaced = flag_misplaced_objects(ref_centers, centers)

        # Draw results
        output_img = draw_boxes(img2.copy(), detections, misplaced)
        cv2.imwrite('output_image.jpg', output_img)
        print("Image processing complete. Output saved as 'output_image.jpg'.")
    else:
        print("Error: Please provide a second image or a video file.")
