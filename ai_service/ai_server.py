from flask import Flask, Response
from flask_cors import CORS
import cv2
import time
import requests
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load your trained model for known faces
model = YOLO("best1_9.pt")  # New model from training epoch 138

# Known person classes from your model
KNOWN_PERSONS = ['aum', 'prem', 'auto']

# Confidence threshold for known person detection
KNOWN_THRESHOLD = 0.6

# Face detection using OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

NODE_API = "http://localhost:3000/api/incident"

# Smart Alert Tracking
class AlertTracker:
    def __init__(self):
        self.active_unknowns = {}  # {location_key: first_seen_time}
        self.known_persons_log = {}  # {name: last_logged_time}
        self.alerted_unknowns = set()  # Locations already alerted
        self.reminder_sent = {}  # {location_key: True} if reminder already sent
        self.ABSENCE_THRESHOLD = 5  # Seconds without detection = person left
        self.REMINDER_TIME = 300  # 5 minutes reminder
        
    def update_unknown(self, location_key, current_time):
        """Track unknown person presence"""
        alert_type = None
        
        if location_key not in self.active_unknowns:
            # NEW unknown person detected
            self.active_unknowns[location_key] = current_time
            
            if location_key not in self.alerted_unknowns:
                # First time seeing this unknown - ALERT!
                alert_type = "entry"
                self.alerted_unknowns.add(location_key)
                self.reminder_sent[location_key] = False
        else:
            # Existing unknown - check for reminder
            first_seen = self.active_unknowns[location_key]
            duration = current_time - first_seen
            
            if duration >= self.REMINDER_TIME and not self.reminder_sent.get(location_key, False):
                # 5 minutes passed - send reminder
                alert_type = "reminder"
                self.reminder_sent[location_key] = True
        
        return alert_type
    
    def cleanup_stale(self, current_time, active_locations):
        """Remove unknowns that have left (not seen recently)"""
        stale_keys = []
        for key, first_seen in list(self.active_unknowns.items()):
            if key not in active_locations:
                # This unknown is no longer being detected
                stale_keys.append(key)
        
        for key in stale_keys:
            del self.active_unknowns[key]
            if key in self.alerted_unknowns:
                self.alerted_unknowns.remove(key)  # Allow re-alert if they return
            if key in self.reminder_sent:
                del self.reminder_sent[key]
    
    def should_log_known(self, name, current_time, cooldown=300):
        """Rate limit known person logging"""
        if (current_time - self.known_persons_log.get(name, 0)) > cooldown:
            self.known_persons_log[name] = current_time
            return True
        return False

# Global tracker
tracker = AlertTracker()

def send_alert(alert_data):
    """Send alert to backend"""
    try:
        requests.post(NODE_API, json=alert_data, timeout=2)
        return True
    except:
        return False

def generate_frames():
    global tracker
    cap = cv2.VideoCapture(0) 
    frame_cnt = 0
    last_cleanup = time.time()
    
    # Colors (BGR format)
    COLOR_UNKNOWN = (0, 0, 255)    # Red for unknown
    person_colors = {
        'aum': (0, 165, 255),      # Orange
        'prem': (0, 255, 0),       # Green
        'auto': (255, 255, 0),     # Cyan
    }

    while True:
        success, frame = cap.read()
        if not success: 
            break

        frame_cnt += 1
        current_time = time.time()
        
        if frame_cnt % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Stage 1: Detect ALL faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            # Stage 2: YOLO for known persons
            results = model(frame, imgsz=320, verbose=False)
            
            identified_faces = []
            active_unknown_locations = set()
            
            # Process known persons
            for r in results:
                for box in r.boxes:
                    name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if conf > KNOWN_THRESHOLD and name.lower() in KNOWN_PERSONS:
                        color = person_colors.get(name.lower(), (0, 255, 0))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} {conf:.0%}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw+4, y1), color, -1)
                        cv2.putText(frame, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                        
                        identified_faces.append((x1, y1, x2, y2))
                        
                        # Log known person (rate limited)
                        if tracker.should_log_known(name, current_time):
                            send_alert({
                                "class": name,
                                "camera": "CAM-01",
                                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                "type": "known"
                            })
                            print(f"✅ Known: {name}")
            
            # Process unknown faces
            for (fx, fy, fw, fh) in faces:
                face_cx, face_cy = fx + fw//2, fy + fh//2
                
                is_known = any(x1 <= face_cx <= x2 and y1 <= face_cy <= y2 
                              for (x1, y1, x2, y2) in identified_faces)
                
                if not is_known:
                    location_key = f"{fx//80}_{fy//80}"  # Grid-based location
                    active_unknown_locations.add(location_key)
                    
                    # Smart alert check
                    alert_type = tracker.update_unknown(location_key, current_time)
                    
                    # Draw red box
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), COLOR_UNKNOWN, 3)
                    label = "UNKNOWN"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (fx, fy-lh-10), (fx+lw+4, fy), COLOR_UNKNOWN, -1)
                    cv2.putText(frame, label, (fx+2, fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    
                    # Send alert based on type
                    if alert_type == "entry":
                        send_alert({
                            "class": "Unknown",
                            "camera": "CAM-01",
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "type": "unknown_entry"
                        })
                        print(f"🚨 UNKNOWN ENTERED!")
                    elif alert_type == "reminder":
                        send_alert({
                            "class": "Unknown (5min)",
                            "camera": "CAM-01",
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "type": "unknown_reminder"
                        })
                        print(f"⚠️ UNKNOWN still present (5 min)")
            
            # Cleanup stale unknowns every 2 seconds
            if current_time - last_cleanup > 2:
                tracker.cleanup_stale(current_time, active_unknown_locations)
                last_cleanup = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)