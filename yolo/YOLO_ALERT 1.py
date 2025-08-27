import cv2
import numpy as np
import time
import datetime
import smtplib
from email.mime.text import MIMEText
import winsound  # For Windows beep sound

# Email configuration
EMAIL_ADDRESS = "your_email@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "your_app_password"   # Use App Password if using Gmail
RECIPIENT_EMAIL = "carljosephbornrich@gmail.com"

class UnattendedObjectDetector:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Load YOLO
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Color map for different objects (BGR format)
        self.color_map = {
            "person": (0, 255, 0),    # Green
            "car": (255, 0, 0),       # Blue
            "chair": (0, 255, 255),   # Yellow
            "bottle": (255, 255, 0),  # Cyan
            "default": (0, 0, 255)    # Red
        }
        
        self.detections = {}
        self.unattended_threshold = 10  # 10 seconds
        self.email_sent = {}

    def send_email_alert(self, object_name, location):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = MIMEText(f"Unattended {object_name} detected!\n"
                          f"Location: {location}\n"
                          f"Time: {timestamp}")
            msg['Subject'] = 'Unattended Object Alert'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECIPIENT_EMAIL

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"Email alert sent for {object_name}")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def detect_objects(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            current_detections = {}

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    object_name = self.classes[class_ids[i]]
                    object_id = f"{object_name}_{x}_{y}"  # Unique ID based on position
                    
                    # Assign color based on object type
                    color = self.color_map.get(object_name, self.color_map["default"])
                    
                    # Track detection time
                    if object_id not in self.detections:
                        self.detections[object_id] = {
                            "time": time.time(),
                            "name": object_name,
                            "location": f"X:{x}, Y:{y}"
                        }
                    
                    elapsed_time = time.time() - self.detections[object_id]["time"]
                    
                    # Draw bounding box and label
                    label = f"{object_name} ({elapsed_time:.1f}s)"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)

                    # Check for unattended status
                    if elapsed_time >= self.unattended_threshold:
                        unattended_label = f"UNATTENDED {object_name}!"
                        cv2.putText(frame, unattended_label, (x, y - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Beep alert
                        winsound.Beep(1000, 500)
                        
                        # Send email if not already sent
                        if object_id not in self.email_sent:
                            self.email_sent[object_id] = self.send_email_alert(
                                object_name, self.detections[object_id]["location"]
                            )

                    current_detections[object_id] = True

            # Remove old detections
            self.detections = {k: v for k, v in self.detections.items() 
                             if k in current_detections}

            cv2.imshow('Unattended Object Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = UnattendedObjectDetector()
    detector.detect_objects()