#!/usr/bin/env python3
"""
NPU Inference Test for RDK X5
Tests YOLOv8 on BPU (NPU) for object detection
"""
import numpy as np
import time
import cv2

# Must source /opt/tros/humble/setup.bash before running
from hobot_dnn import pyeasy_dnn

# Pre-installed YOLOv8 model
MODEL_PATH = "/opt/hobot/model/x5/basic/yolov8_640x640_nv12.bin"

def bgr_to_nv12(bgr_img, target_size=(640, 640)):
    """Convert BGR image to NV12 format required by BPU"""
    # Resize to model input size
    img_resized = cv2.resize(bgr_img, target_size)
    
    # Convert BGR to YUV
    yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420)
    
    # Convert YUV I420 to NV12
    h, w = target_size[1], target_size[0]
    y = yuv[:h, :]
    u = yuv[h:h + h//4, :].reshape(h//2, w//2)
    v = yuv[h + h//4:, :].reshape(h//2, w//2)
    
    # Interleave UV
    uv = np.zeros((h//2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    
    nv12 = np.vstack((y, uv))
    return nv12

def main():
    print("=" * 50)
    print("RDK X5 NPU Inference Test")
    print("=" * 50)
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    try:
        models = pyeasy_dnn.load(MODEL_PATH)
        model = models[0]
        print(f"✅ Model loaded: {model.name}")
        print(f"   Estimated latency: {model.estimate_latency}ms")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Print model info
    print(f"\n📊 Model inputs:")
    for i, inp in enumerate(model.inputs):
        print(f"   [{i}] shape={inp.properties.shape}, type={inp.properties.tensor_type}")
    
    print(f"\n📊 Model outputs:")
    for i, out in enumerate(model.outputs):
        print(f"   [{i}] shape={out.properties.shape}")
    
    # Test with camera
    print("\n📷 Opening camera...")
    cap = cv2.VideoCapture(8)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        cap.release()
        return
    
    print(f"✅ Camera OK, frame shape: {frame.shape}")
    
    # Convert to NV12
    print("\n🔄 Converting to NV12...")
    nv12_data = bgr_to_nv12(frame)
    print(f"   NV12 shape: {nv12_data.shape}")
    
    # Run inference
    print("\n🚀 Running NPU inference...")
    
    # Benchmark
    times = []
    for i in range(10):
        nv12_data = bgr_to_nv12(frame)
        start = time.time()
        outputs = model.forward(nv12_data)
        end = time.time()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    
    print(f"\n📈 Benchmark Results:")
    print(f"   Average inference time: {avg_time:.2f}ms")
    print(f"   Estimated FPS: {fps:.1f}")
    print(f"   Output tensors: {len(outputs)}")
    
    cap.release()
    print("\n✅ NPU test complete!")

if __name__ == "__main__":
    main()
