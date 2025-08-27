import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # Pretrained segmentation model

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)
    
    # Initialize masks and overlay
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    overlay = frame.copy()

    if results[0].masks is not None:
        for mask in results[0].masks:
            # Process mask
            mask_data = mask.data[0].cpu().numpy()
            resized_mask = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
            _, binary_mask = cv2.threshold(resized_mask, 0.5, 1, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)

            # Update combined mask for edge detection
            combined_mask = cv2.bitwise_or(combined_mask, binary_mask)

            # Add segmentation overlay (blue)
            overlay[binary_mask == 1] = (255, 0, 0)

    # Apply semi-transparent segmentation overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Edge detection processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    masked_edges = cv2.bitwise_and(edges, edges, mask=combined_mask)

    # Create green edge overlay
    edge_overlay = np.zeros_like(frame)
    edge_overlay[:, :, 1] = masked_edges  # Green channel

    # Combine edge overlay with frame
    frame = cv2.add(frame, edge_overlay)

    # Display results
    cv2.imshow('Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()