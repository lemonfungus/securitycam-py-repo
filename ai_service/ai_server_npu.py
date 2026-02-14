#!/usr/bin/env python3
"""
AI Security Camera Server - RDK X5 NPU Version
Uses Horizon hobot_dnn for 10 TOPS NPU acceleration
"""
import os
from flask import Flask, Response
from flask_cors import CORS
import cv2
import time
import requests
import numpy as np

# Must have sourced /opt/tros/humble/setup.bash before running
from hobot_dnn import pyeasy_dnn

app = Flask(__name__)
CORS(app)

# Camera source (video8 for Pi Camera on RDK X5)
CAMERA_SOURCE = int(os.environ.get('CAMERA_SOURCE', '8'))

# NPU Model path (pre-installed YOLOv8)
NPU_MODEL_PATH = "/opt/hobot/model/x5/basic/yolov8_640x640_nv12.bin"

# COCO class names for YOLOv8
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

NODE_API = "http://localhost:3000/api/incident"

# Load NPU model
print(f"Loading NPU model: {NPU_MODEL_PATH}")
models = pyeasy_dnn.load(NPU_MODEL_PATH)
npu_model = models[0]
print(f"✅ Model loaded: {npu_model.name}, Latency: {npu_model.estimate_latency}ms")

def bgr_to_nv12(bgr_img, target_size=(640, 640)):
    """Convert BGR image to NV12 format for BPU"""
    img_resized = cv2.resize(bgr_img, target_size)
    yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420)
    h, w = target_size[1], target_size[0]
    y = yuv[:h, :]
    u = yuv[h:h + h//4, :].reshape(h//2, w//2)
    v = yuv[h + h//4:, :].reshape(h//2, w//2)
    uv = np.zeros((h//2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.vstack((y, uv))

def postprocess_yolov8(outputs, orig_shape, conf_thresh=0.5, nms_thresh=0.45):
    """Post-process YOLOv8 output to get bounding boxes"""
    # YOLOv8 output shape: [1, 84, 8400] -> transpose to [8400, 84]
    output = outputs[0].buffer
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dimension
    
    # Transpose if needed
    if output.shape[0] == 84:
        output = output.T  # [8400, 84]
    
    # Split into boxes and class scores
    boxes = output[:, :4]  # cx, cy, w, h
    scores = output[:, 4:]  # 80 class scores
    
    # Get max class score and class id for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter by confidence
    mask = confidences > conf_thresh
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # Convert from cx,cy,w,h to x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # Scale to original image size
    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    x1 = (x1 * scale_x).astype(int)
    y1 = (y1 * scale_y).astype(int)
    x2 = (x2 * scale_x).astype(int)
    y2 = (y2 * scale_y).astype(int)
    
    # Simple NMS
    detections = []
    indices = np.argsort(confidences)[::-1]
    
    for i in indices[:20]:  # Limit to top 20
        detections.append({
            'box': (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
            'class': COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else 'unknown',
            'conf': float(confidences[i])
        })
    
    return detections

def send_alert(alert_data):
    """Send alert to backend"""
    try:
        requests.post(NODE_API, json=alert_data, timeout=2)
    except:
        pass

last_alert_time = {}

def generate_frames():
    """Generate video frames with NPU inference"""
    global last_alert_time
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_cnt = 0
    fps_start = time.time()
    fps_count = 0
    current_fps = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_cnt += 1
        fps_count += 1
        
        # Calculate FPS every second
        if time.time() - fps_start >= 1.0:
            current_fps = fps_count
            fps_count = 0
            fps_start = time.time()
        
        # Run inference every 2 frames for balance
        if frame_cnt % 2 == 0:
            # Convert to NV12 for NPU
            nv12_data = bgr_to_nv12(frame)
            
            # NPU inference
            outputs = npu_model.forward(nv12_data)
            
            # Post-process
            detections = postprocess_yolov8(outputs, frame.shape)
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['box']
                cls = det['class']
                conf = det['conf']
                
                # Color based on class
                if cls == 'person':
                    color = (0, 255, 0)  # Green
                    
                    # Alert for person detection (rate limited)
                    current_time = time.time()
                    if current_time - last_alert_time.get('person', 0) > 60:
                        send_alert({
                            "class": "Person",
                            "camera": "CAM-01",
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "type": "person_detected"
                        })
                        last_alert_time['person'] = current_time
                else:
                    color = (255, 165, 0)  # Orange
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls} {conf:.0%}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        cv2.putText(frame, f"NPU FPS: {current_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {'status': 'ok', 'model': npu_model.name, 'latency_ms': npu_model.estimate_latency}

if __name__ == '__main__':
    print("🚀 Starting NPU-accelerated AI Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
