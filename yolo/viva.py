import cv2
import time
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import numpy as np
import winsound  # For Windows beep sound
from ultralytics import YOLO
import os

# Email configuration
EMAIL_ADDRESS = "carljosephbornrich@gmail.com"
EMAIL_PASSWORD = "wyvm jfnk anwz xyvo"  # Replace with your Gmail App Password
RECIPIENT_EMAIL = "carljosephbornrich@gmail.com"

class UnattendedObjectDetector:
    def __init__(self):
        try:
            self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.model = YOLO("yolov8n.pt")  # Try 'yolov8s.pt' for better accuracy
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Initialization error: {e}")
            exit(1)
        self.unattended_threshold = 10  # 10 seconds
        self.tracked_objects = {}  # {id: (bbox, class_name, start_time, email_sent, smoothed_bbox, miss_count)}
        self.snapshot_path = "unattended_object.jpg"
        self.debug_path = "debug_missed_frames"
        os.makedirs(self.debug_path, exist_ok=True)
        self.target_classes = ["handbag", "backpack", "suitcase", "cell phone"]
        self.object_id_counter = 0
        self.conf_threshold = 0.15  # Lowered for stationary objects
        self.max_miss_frames = 10  # Increased persistence
        self.frame_count = 0

    def send_email_alert(self, frame, x, y, w, h, object_class):
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f'Unattended {object_class} Alert'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECIPIENT_EMAIL

            location = f"Location: x={x}, y={y}, width={w}, height={h}"
            body = (f"Unattended {object_class} detected at "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"{location}")
            msg.attach(MIMEText(body, 'plain'))

            cv2.imwrite(self.snapshot_path, frame)
            with open(self.snapshot_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=self.snapshot_path)
                msg.attach(img)

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"Email alert sent for {object_class} at {location}")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def smooth_bbox(self, prev_bbox, new_bbox, alpha=0.7):
        if prev_bbox is None:
            return new_bbox
        px, py, pw, ph = prev_bbox
        nx, ny, nw, nh = new_bbox
        return (
            int(alpha * px + (1 - alpha) * nx),
            int(alpha * py + (1 - alpha) * ny),
            int(alpha * pw + (1 - alpha) * nw),
            int(alpha * ph + (1 - alpha) * nh)
        )

    def is_stationary(self, prev_bbox, new_bbox, threshold=10):
        if prev_bbox is None:
            return True
        px, py, _, _ = prev_bbox
        nx, ny, _, _ = new_bbox
        distance = np.sqrt((nx - px) ** 2 + (ny - py) ** 2)
        return distance < threshold

    def detect_unattended_object(self):
        last_frame_time = time.time()
        while True:
            self.frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break

            # Limit frame rate to 10 FPS
            if time.time() - last_frame_time < 0.1:
                continue
            last_frame_time = time.time()

            # Preprocess frame: enhance contrast (CLAHE)
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(frame_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit
            l = clahe.apply(l)
            frame_lab = cv2.merge((l, a, b))
            frame_processed = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
            frame_display = frame.copy()  # Keep original frame for display

            # Run YOLOv8 inference
            try:
                results = self.model(frame_processed, verbose=False, imgsz=640)
            except Exception as e:
                print(f"YOLO inference failed: {e}")
                continue

            current_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name in self.target_classes:
                        x, y, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x, y2 - y
                        conf = float(box.conf[0])
                        if conf > self.conf_threshold:
                            current_objects.append((x, y, w, h, class_name, conf))
                            print(f"Frame {self.frame_count}: Detected {class_name} at ({x}, {y}, {w}, {h}) with confidence {conf:.2f}")
                            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame_display, f"{class_name} {conf:.2f}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save frame if no objects detected
            if not current_objects:
                print(f"Frame {self.frame_count}: No target objects detected")
                cv2.imwrite(os.path.join(self.debug_path, f"missed_frame_{self.frame_count}.jpg"), frame_display)

            # Track objects
            new_tracked_objects = {}
            for obj in current_objects:
                x, y, w, h, class_name, conf = obj
                matched = False
                for oid, (prev_bbox, prev_class, start_time, email_sent, smoothed_bbox, miss_count) in self.tracked_objects.items():
                    if self.iou((x, y, w, h), prev_bbox) > 0.6 and class_name == prev_class:
                        smoothed_bbox = self.smooth_bbox(smoothed_bbox, (x, y, w, h))
                        is_stat = self.is_stationary(smoothed_bbox, (x, y, w, h))
                        new_tracked_objects[oid] = (smoothed_bbox, class_name, start_time, email_sent, smoothed_bbox, 0)
                        matched = True
                        elapsed_time = time.time() - start_time
                        status = f"{class_name} ID:{oid} Time:{elapsed_time:.1f}s {'Stationary' if is_stat else 'Moving'}"
                        sx, sy, sw, sh = smoothed_bbox
                        cv2.rectangle(frame_display, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
                        cv2.putText(frame_display, status, (sx, sy + sh + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if elapsed_time >= self.unattended_threshold and not email_sent and is_stat:
                            cv2.putText(frame_display, f"UNATTENDED {class_name.upper()}!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            winsound.Beep(1000, 500)
                            email_sent = self.send_email_alert(frame_display, sx, sy, sw, sh, class_name)
                            new_tracked_objects[oid] = (smoothed_bbox, class_name, start_time, email_sent, smoothed_bbox, 0)
                        break
                if not matched:
                    smoothed_bbox = (x, y, w, h)
                    new_tracked_objects[self.object_id_counter] = ((x, y, w, h), class_name, time.time(), False, smoothed_bbox, 0)
                    self.object_id_counter += 1

            # Update miss counts for unmatched objects
            for oid, (prev_bbox, prev_class, start_time, email_sent, smoothed_bbox, miss_count) in self.tracked_objects.items():
                if oid not in new_tracked_objects:
                    miss_count += 1
                    if miss_count <= self.max_miss_frames:
                        new_tracked_objects[oid] = (prev_bbox, prev_class, start_time, email_sent, smoothed_bbox, miss_count)
                        sx, sy, sw, sh = smoothed_bbox
                        elapsed_time = time.time() - start_time
                        status = f"{prev_class} ID:{oid} Time:{elapsed_time:.1f}s Missed:{miss_count}"
                        cv2.rectangle(frame_display, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                        cv2.putText(frame_display, status, (sx, sy + sh + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        if elapsed_time >= self.unattended_threshold and not email_sent:
                            cv2.putText(frame_display, f"UNATTENDED {prev_class.upper()}!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            winsound.Beep(1000, 500)
                            email_sent = self.send_email_alert(frame_display, sx, sy, sw, sh, prev_class)
                            new_tracked_objects[oid] = (prev_bbox, prev_class, start_time, email_sent, smoothed_bbox, miss_count)

            self.tracked_objects = new_tracked_objects
            if not current_objects and all(info[5] >= self.max_miss_frames for info in self.tracked_objects.values()):
                self.tracked_objects = {}

            # Display frame
            cv2.imshow('Unattended Object Detector', frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = UnattendedObjectDetector()
        detector.detect_unattended_object()
    except Exception as e:
        print(f"Program failed: {e}")