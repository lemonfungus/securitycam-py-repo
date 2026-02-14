"""
Hybrid Face Recognition on RDK X5
- Detection: MediaPipe (CPU) - Robust & Easy
- Recognition: ArcFace (BPU) - Fast & Accurate
"""
import cv2
import numpy as np
import os
import time
import threading
import sys
from flask import Flask, Response, jsonify

# Try to import MediaPipe
try:
    import mediapipe as mp
except ImportError:
    print("Error: mediapipe not found! Run 'pip install mediapipe'")
    sys.exit(1)

# Import BPU library for Recognition only
try:
    from bpu_face import ArcFace_BPU
except ImportError:
    print("Error: bpu_face.py not found!")
    sys.exit(1)

# Configuration
KNOWN_FACES_DIR = "known_faces"
MODELS_DIR = "/home/sunrise/models"
REC_MODEL = os.path.join(MODELS_DIR, "face_id_r50.bin")

CAMERA_ID = 8
REC_THRESHOLD = 0.85
FLIP_180 = True  # Keep true for upside down camera

app_flask = Flask(__name__)
latest_frame = None
latest_results = []
lock = threading.Lock()

# Init MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def load_known_faces(rec_model):
    """Load known face embeddings."""
    known_faces = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        return known_faces

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Use MediaPipe to find the face (Robust to masks)
            # Convert to RGB
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            
            if results.detections:
                # Take the largest face
                best_det = results.detections[0] # Usually only 1 face in ref photo
                bboxC = best_det.location_data.relative_bounding_box
                x = max(0, int(bboxC.xmin * w))
                y = max(0, int(bboxC.ymin * h))
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                
                face_crop = img[y:y2, x:x2]
                if face_crop.size > 0:
                    emb = rec_model.get_embedding(face_crop)
                    if emb is not None:
                        embeddings.append(emb)
            else:
                print(f"Warning: No face detected in {img_name}")

        if embeddings:
            known_faces[person_name] = embeddings
            print(f"  Loaded {len(embeddings)} faces for {person_name}")
        else:
            print(f"  Warning: 0 faces loaded for {person_name}!")
    
    return known_faces

def recognize_face(embedding, known_faces):
    best_match = None
    best_score = 0
    for name, emb_list in known_faces.items():
        for known_emb in emb_list:
            score = np.dot(embedding, known_emb)
            if score > best_score and score > REC_THRESHOLD:
                best_score = score
                best_match = name
    return best_match, best_score

def camera_loop(rec_model, known_faces):
    global latest_frame, latest_results
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error opening camera")
        return
        
    print("Camera opened. Starting MediaPipe + BPU loop...")
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        if FLIP_180:
            frame = cv2.flip(frame, -1)
            
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. MediaPipe Detection (CPU)
        results_mp = face_detection.process(rgb_frame)
        
        final_results = []
        
        if results_mp.detections:
            for detection in results_mp.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                
                # Expand box slightly for ArcFace
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                
                # 2. BPU Recognition
                # ArcFace needs 5 points (kps) usually.
                # MediaPipe gives 6 keypoints.
                # Mapping MP keypoints to ArcFace alignment is possible.
                # MP: 0:RightEye, 1:LeftEye, 2:NoseTip, 3:MouthCenter, 4:RightEar, 5:LeftEar
                # ArcFace expects: LeftEye, RightEye, Nose, LeftMouth, RightMouth (usually)
                # Or we can just crop and resize (simpler but less accurate).
                # BPU ArcFace usually handles alignment internally if we pass 5 points?
                # ArcFace_BPU.get_embedding(img, kps)
                # Let's try to map MP points.
                
                kps = []
                for kp in detection.location_data.relative_keypoints:
                    kps.append([kp.x * w, kp.y * h])
                
                # MP order: RE, LE, Nose, Mouth, REar, LEar
                # ArcFace wants: LE, RE, Nose, LM, RM
                # We have MouthCenter (3).
                # We lack LM, RM.
                # Simplest: Pass None for kps and let ArcFace center crop?
                # Or implement simple alignment.
                # Let's just create a dummy kps compatible with `bpu_face.py` logic
                # Actually `bpu_face.py` uses `face_align.norm_crop(img, kps)`.
                # If kps is None, we need to handle it.
                # Let's verify `bpu_face.py`:
                # def get_embedding(self, img, kps=None):
                # ... if kps is not None: ... aligned = norm_crop ...
                # ... else: aligned = cv2.resize(img, (112, 112))
                
                # So passing None works (Resize). This is robust enough for now.
                
                face_img = frame[y:y2, x:x2]
                if face_img.size == 0: continue
                
                emb = rec_model.get_embedding(face_img) # Pass cropped face
                
                name = "Unknown"
                score = 0.0
                if emb is not None:
                    name, score = recognize_face(emb, known_faces)
                    
                final_results.append({
                    "bbox": [x, y, x2, y2],
                    "name": name,
                    "score": score
                })
        
        with lock:
            latest_results = final_results
            
        # Draw
        if frame_count % 2 == 0:
            draw_frame = frame.copy()
            for r in latest_results:
                x1, y1, x2, y2 = r["bbox"]
                name = r["name"]
                score = r["score"]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_frame, f"{name} {score:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            with lock:
                latest_frame = draw_frame
        
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            print(f"FP: {fps:.1f}, Faces: {len(latest_results)}")

def generate_frames():
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)

@app_flask.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app_flask.route('/')
def index():
    return "<h1>Hybrid Face Rec (MediaPipe + BPU)</h1><img src='/video_feed' />"

if __name__ == "__main__":
    print("Loading BPU Rec Model...")
    rec_model = ArcFace_BPU(REC_MODEL)
    print("Loading Faces...")
    known_faces = load_known_faces(rec_model)
    print(f"Loaded {len(known_faces)} people.")
    
    t = threading.Thread(target=camera_loop, args=(rec_model, known_faces), daemon=True)
    t.start()
    
    app_flask.run(host='0.0.0.0', port=5000, threaded=True)
