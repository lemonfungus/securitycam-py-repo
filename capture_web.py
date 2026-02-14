"""
Web Capture Tool for RDK X5
- Flask Server to stream camera.
- Web UI to enter Name and Click Capture Buttons (Normal, Mask, Side).
- Saves images to `captured_dataset/`.
"""
import cv2
import time
import os
import threading
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

CAMERA_ID = 8
BASE_DIR = "captured_dataset"
FLIP_180 = True

# Thread-safe Camera Handling
outputFrame = None
lock = threading.Lock()

def start_camera_thread():
    t = threading.Thread(target=read_camera_loop)
    t.daemon = True
    t.start()

def read_camera_loop():
    global outputFrame, lock
    cap = cv2.VideoCapture(CAMERA_ID)
    # Don't set resolution (Driver might hang)
    # Default is likely 1920x1080
    
    # Limit Camera FPS to 30
    while True:
        success, frame = cap.read()
        if success:
            if FLIP_180:
                frame = cv2.flip(frame, -1)
                
            with lock:
                outputFrame = frame.copy()
        time.sleep(0.033)

def gen_frames():
    global outputFrame, lock
    while True:
        frame_bytes = None
        local_frame = None
        
        with lock:
            if outputFrame is None:
                time.sleep(0.1)
                continue
            local_frame = outputFrame.copy()
            
        if local_frame is not None:
            # Resize for Preview (Save Bandwidth)
            # Full Res is likely 1080p (Heavy), Resize to 640px wide
            h, w = local_frame.shape[:2]
            aspect = w / h
            new_w = 640
            new_h = int(new_w / aspect)
            local_frame = cv2.resize(local_frame, (new_w, new_h))
            
            ret, buffer = cv2.imencode('.jpg', local_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame_bytes = buffer.tobytes()
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit Preview FPS to 15 (Smooth enough, low lag)
        time.sleep(0.066)

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>RDK Capture Tool</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; text-align: center; background: #111; color: #fff; }
        img { width: 100%; max-width: 640px; border: 2px solid #444; }
        .controls { margin: 20px; padding: 20px; background: #222; border-radius: 10px; display: inline-block; }
        input { padding: 10px; font-size: 16px; width: 200px; text-align: center; }
        button { padding: 10px 20px; font-size: 16px; margin: 5px; cursor: pointer; border: none; border-radius: 5px; color: white; }
        .btn-normal { background: #28a745; }
        .btn-mask { background: #ffc107; color: black; }
        .btn-side { background: #17a2b8; }
        .btn-up { background: #6c757d; }
        .log { margin-top: 10px; color: #0f0; }
    </style>
</head>
<body>
    <h1>Face Data Collector 📸</h1>
    <div class="controls">
        <input type="text" id="name" placeholder="Enter Name (e.g. Prem)" value="Prem">
        <br><br>
        <button class="btn-normal" onclick="capture('normal')">Capture Normal</button>
        <button class="btn-mask" onclick="capture('mask')">Capture Mask</button>
        <button class="btn-side" onclick="capture('side')">Capture Side</button>
        <button class="btn-up" onclick="capture('up_down')">Capture Up/Down</button>
        <div class="log" id="log">Ready...</div>
    </div>
    <br>
    <img src="{{ url_for('video_feed') }}">

    <script>
        function capture(mode) {
            var name = document.getElementById('name').value;
            if(!name) { alert("Please enter a name!"); return; }
            
            document.getElementById('log').innerText = "Capturing " + mode + "...";
            
            fetch('/capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name, mode: mode})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('log').innerText = data.message;
            })
            .catch(err => {
                 document.getElementById('log').innerText = "Error: " + err;
            });
        }
    </script>
</body>
</html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global outputFrame, lock
    data = request.get_json()
    name = data.get('name', 'Unknown')
    mode = data.get('mode', 'normal')
    
    frame = None
    with lock:
        if outputFrame is not None:
            frame = outputFrame.copy()
            
    if frame is not None:
        save_dir = os.path.join(BASE_DIR, name, mode)
        os.makedirs(save_dir, exist_ok=True)
        
        ts = int(time.time() * 1000)
        filename = f"{name}_{mode}_{ts}.jpg"
        path = os.path.join(save_dir, filename)
        
        cv2.imwrite(path, frame)
        return jsonify({"message": f"Saved {filename} to {mode}"})
    
    return jsonify({"message": "Capture Failed! No Frame."}), 500

if __name__ == '__main__':
    # Create Base Dir
    os.makedirs(BASE_DIR, exist_ok=True)
    # Start Camera Thread
    start_camera_thread()
    app.run(host='0.0.0.0', port=5000, threaded=True)
