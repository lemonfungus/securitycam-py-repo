"""
Two-Stage Face Recognition Demo using InsightFace
Stage 1: Face detection (RetinaFace)
Stage 2: Face recognition (ArcFace)
"""
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import defaultdict

# Configuration
KNOWN_FACES_DIR = "known_faces"
TEST_IMAGES_DIR = r"Trainscript\model3.v5i.yolov8\train\images"
LABELS_DIR = r"Trainscript\model3.v5i.yolov8\train\labels"

def load_known_faces(app):
    """Load and encode all known faces from the database."""
    known_faces = {}
    
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            faces = app.get(img)
            if len(faces) > 0:
                embeddings.append(faces[0].embedding)
        
        if embeddings:
            # Average embedding for this person
            known_faces[person_name] = np.mean(embeddings, axis=0)
            print(f"  Loaded {len(embeddings)} faces for {person_name}")
    
    return known_faces

def recognize_face(embedding, known_faces, threshold=0.3):
    """Find the best matching face from known faces."""
    best_match = None
    best_score = 0
    
    for name, known_embedding in known_faces.items():
        # Cosine similarity
        score = np.dot(embedding, known_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
        )
        if score > best_score and score > threshold:
            best_score = score
            best_match = name
    
    return best_match, best_score

def get_ground_truth(label_path, class_map):
    """Get ground truth class from YOLO label file."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if lines:
            class_id = int(lines[0].split()[0])
            return class_map.get(class_id, None)
    return None

def main():
    # Class mapping from data.yaml
    class_map = {0: 'Aum', 1: 'Auto', 2: 'Jiw', 3: 'Prem'}
    
    print("Loading InsightFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("\nLoading known faces...")
    known_faces = load_known_faces(app)
    print(f"Loaded {len(known_faces)} people: {list(known_faces.keys())}")
    
    # Get ALL test images
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\nTesting on {len(test_images)} images...")
    
    # Track results
    results = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    no_face = 0
    
    for img_name in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        
        # Get ground truth
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(LABELS_DIR, label_name)
        ground_truth = get_ground_truth(label_path, class_map)
        
        faces = app.get(img)
        
        if len(faces) == 0:
            no_face += 1
            continue
        
        for face in faces:
            embedding = face.embedding
            predicted, score = recognize_face(embedding, known_faces)
            
            if predicted is None:
                predicted = "Unknown"
            
            # Track for confusion matrix
            if ground_truth:
                results[ground_truth][predicted] += 1
                if predicted == ground_truth:
                    correct += 1
                total += 1
    
    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total images: {len(test_images)}")
    print(f"Faces detected: {total}")
    print(f"No face detected: {no_face}")
    print(f"Correct predictions: {correct}/{total} = {100*correct/total:.1f}%")
    
    print(f"\n{'='*50}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*50}")
    print(f"{'True':<10} -> Predictions")
    for true_class in ['Aum', 'Auto', 'Jiw', 'Prem']:
        preds = results[true_class]
        total_class = sum(preds.values())
        if total_class > 0:
            pred_str = ", ".join([f"{k}:{v}({100*v/total_class:.0f}%)" for k,v in preds.items()])
            print(f"{true_class:<10} -> {pred_str}")

if __name__ == "__main__":
    main()
