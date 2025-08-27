import cv2
import time
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import numpy as np
import winsound  # For Windows beep sound

# Email configuration
EMAIL_ADDRESS = "carljosephbornrich@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "wyvm jfnk anwz xyvo"   # Use App Password if using Gmail9
RECIPIENT_EMAIL = "carljosephbornrich@gmail.com"

class UnattendedObjectDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.object_detected_time = None
        self.unattended_threshold = 10  # 10 seconds
        self.email_sent = False
        self.snapshot_path = "unattended_object.jpg"

    def send_email_alert(self, frame, x, y, w, h):
        try:
            # Create a multipart message
            msg = MIMEMultipart()
            msg['Subject'] = 'Unattended/Misplaced Object Alert'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECIPIENT_EMAIL

            # Email body with location
            location = f"Location: x={x}, y={y}, width={w}, height={h}"
            body = (f"Unattended/misplaced object detected at "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"{location}")
            msg.attach(MIMEText(body, 'plain'))

            # Save and attach the image
            cv2.imwrite(self.snapshot_path, frame)
            with open(self.snapshot_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=self.snapshot_path)
                msg.attach(img)

            # Send the email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            print("Email alert sent successfully with image")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def detect_unattended_object(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Noise reduction
            fg_mask = cv2.medianBlur(fg_mask, 5)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Adjust threshold as needed
                    object_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Track unattended time
                    if self.object_detected_time is None:
                        self.object_detected_time = time.time()
                    else:
                        elapsed_time = time.time() - self.object_detected_time
                        status = f"Time elapsed: {elapsed_time:.1f}s"
                        cv2.putText(frame, status, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Check if object is unattended for 10 seconds
                        if elapsed_time >= self.unattended_threshold:
                            cv2.putText(frame, "UNATTENDED/MISPLACED OBJECT!", (10, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            # Beep alert
                            winsound.Beep(1000, 500)  # 1000 Hz, 500 ms
                            
                            # Send email with location and image if not already sent
                            if not self.email_sent:
                                self.email_sent = self.send_email_alert(frame, x, y, w, h)

                    break  # Process only the largest object

            if not object_detected:
                self.object_detected_time = None
                self.email_sent = False

            # Display the frame
            cv2.imshow('Unattended Object Detector', frame)
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = UnattendedObjectDetector()
    detector.detect_unattended_object()