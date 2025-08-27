from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the nano model for faster performance

# Initialize webcam (use CAP_DSHOW for Windows, comment out if on Linux/Mac)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows
# cap = cv2.VideoCapture(0)  # Use this line instead if on Linux/Mac

# Set the resolution to 720p (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Verify the resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Set resolution to: {width}x{height}")

# Check if the resolution is actually 720p
if width == 1280 and height == 720:
    print("Successfully set to 720p!")
else:
    print("Failed to set 720p. Using default resolution. Check your webcam's supported resolutions.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Check if webcam is connected.")
        break

    # Perform object detection
    results = model(frame)

    # Display the results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()