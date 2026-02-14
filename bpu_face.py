import numpy as np
import cv2
import os
import sys
import time

# Try to import BPU library
BPU_LIB_LOADED = False
try:
    from hobot_dnn import pyeasy_dnn as dnn
    BPU_LIB_LOADED = True
    print("[BPU] Loaded hobot_dnn")
except ImportError:
    print("[BPU] Warning: hobot_dnn not found. Running in CPU simulation mode (slow).")

class BPUModel:
    def __init__(self, model_path, input_shape):
        self.model_path = model_path
        self.h, self.w = input_shape
        self.model = None
        if BPU_LIB_LOADED and os.path.exists(model_path):
            try:
                self.model = dnn.load(model_path)[0]
                print(f"[BPU] Loaded {model_path}")
            except Exception as e:
                print(f"[BPU] Error loading {model_path}: {e}")

    def preprocess(self, img):
        # Resize to model input
        img_resized = cv2.resize(img, (self.w, self.h))
        
        # Convert to NV12
        yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420)
        
        h_y = self.h
        w_y = self.w
        y_size = h_y * w_y
        
        # Y plane
        y_plane = yuv[:h_y, :].flatten()
        
        # U and V planes
        u_plane = yuv[h_y : h_y + h_y // 4, :].flatten()
        v_plane = yuv[h_y + h_y // 4 :, :].flatten()
        
        # Interleave U and V for NV12
        uv_plane = np.zeros(y_size // 2, dtype=np.uint8)
        uv_plane[0::2] = u_plane
        uv_plane[1::2] = v_plane
        
        # Combine
        input_data = np.concatenate((y_plane, uv_plane))
        return input_data

    def forward(self, img):
        if self.model is None:
            return None
        
        t0 = time.time()
        input_data = self.preprocess(img)
        outputs = self.model.forward(input_data)
        
        # Convert BPU tensor outputs to numpy
        numpy_outputs = []
        for out in outputs:
            numpy_outputs.append(out.buffer)
            
        return numpy_outputs


class SCRFD_BPU(BPUModel):
    def __init__(self, model_path, input_shape=(640, 640), conf_thresh=0.15, nms_thresh=0.4):
        super().__init__(model_path, input_shape)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.fmc = 3
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2
        self.center_cache = {}
        
    def forward(self, img):
        return super().forward(img)

    def _get_anchors(self, height, width, stride):
        key = (height, width, stride)
        if key not in self.center_cache:
            y, x = np.mgrid[:height, :width]
            xy = np.stack((x, y), axis=-1).astype(np.float32) * stride
            self.center_cache[key] = xy
        return self.center_cache[key]

    def detect(self, img, input_size=None):
        if self.model is None:
            return []

        # Letterbox
        im_h, im_w = img.shape[:2]
        target_h, target_w = self.h, self.w
        
        scale = min(target_w / im_w, target_h / im_h)
        new_w, new_h = int(im_w * scale), int(im_h * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w, :] = img_resized
        
        outputs = self.forward(canvas)
        if outputs is None:
            return []
            
        structured_outs = [None] * 9
        
        for i, out in enumerate(outputs):
            if i < 3: 
                type_idx = 0; stride_idx = i; channels = 2
            elif i < 6:
                type_idx = 1; stride_idx = i - 3; channels = 8
            else:
                type_idx = 2; stride_idx = i - 6; channels = 20
                
            stride = self.feat_stride_fpn[stride_idx]
            feat_h = self.h // stride
            feat_w = self.w // stride
            
            expected_size = feat_h * feat_w * channels
            arr = np.frombuffer(out, dtype=np.float32)
            
            if arr.size != expected_size:
                continue
                
            arr = arr.reshape((1, feat_h, feat_w, channels))
            target_idx = stride_idx * 3 + type_idx
            structured_outs[target_idx] = arr
            
        if any(x is None for x in structured_outs):
            return []

        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        # Skip Stride 8 (idx=0)
        for idx in range(1, 3):
            base_idx = idx * 3
            scores = structured_outs[base_idx]
            bbox_preds = structured_outs[base_idx + 1]
            kps_preds = structured_outs[base_idx + 2]
            
            stride = self.feat_stride_fpn[idx]
            height, width = scores.shape[1:3]
            
            anchors = self._get_anchors(height, width, stride).reshape(-1, 2)
            
            scores = scores.reshape(-1, 2)
            bbox_preds = bbox_preds.reshape(-1, 2, 4)
            kps_preds = kps_preds.reshape(-1, 2, 10)
            
            # Select Anchor 1 (Idx 0)
            face_scores = scores[:, 0]
            bbox_preds = bbox_preds[:, 0, :]
            kps_preds = kps_preds[:, 0, :]
            
            valid_inds = np.where(face_scores > self.conf_thresh)[0]
            if len(valid_inds) == 0:
                continue
                
            face_scores = face_scores[valid_inds]
            bbox_preds = bbox_preds[valid_inds]
            kps_preds = kps_preds[valid_inds]
            anchors_sel = anchors[valid_inds]
            
            bbox_preds = bbox_preds * stride
            
            x1 = anchors_sel[:, 0] - bbox_preds[:, 0]
            y1 = anchors_sel[:, 1] - bbox_preds[:, 1]
            x2 = anchors_sel[:, 0] + bbox_preds[:, 2]
            y2 = anchors_sel[:, 1] + bbox_preds[:, 3]
            
            # Un-Letterbox
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)
            
            # KPS
            kps_preds = kps_preds * stride
            kps = np.zeros((len(face_scores), 10), dtype=np.float32)
            for k in range(5):
                kx = anchors_sel[:, 0] + kps_preds[:, k*2]
                ky = anchors_sel[:, 1] + kps_preds[:, k*2+1]
                kx = (kx - pad_w) / scale
                ky = (ky - pad_h) / scale
                kps[:, k*2] = kx
                kps[:, k*2+1] = ky
                
            scores_list.append(face_scores)
            bboxes_list.append(bboxes)
            kpss_list.append(kps)
            
        if not scores_list:
            return []
            
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)
        
        keep = self.nms(bboxes, scores, self.nms_thresh)
        
        if len(keep) > 0:
            top_score = scores[keep[0]]
            print(f"[DEBUG] Top Score: {top_score:.4f} | Count: {len(keep)}")
        
        final_dets = []
        for i in keep:
            det = type('Face', (), {})()
            det.bbox = bboxes[i]
            det.kps = kpss[i].reshape(5, 2)
            det.det_score = scores[i]
            det.embedding = None
            final_dets.append(det)
            
        return final_dets

    def nms(self, bboxes, scores, thresh):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

class ArcFace_BPU(BPUModel):
    def __init__(self, model_path, input_shape=(112, 112)):
        super().__init__(model_path, input_shape)
    
    def get_embedding(self, img, kps=None):
        if kps is not None:
             img = self.align_face(img, kps)

        outputs = self.forward(img)
        if outputs:
            raw_buf = outputs[0]
            emb = np.frombuffer(raw_buf, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            return emb
        return None
        
    def align_face(self, img, kps):
        target_kps = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
            
        tform = cv2.estimateAffinePartial2D(kps, target_kps)[0]
        aligned = cv2.warpAffine(img, tform, (112, 112))
        return aligned
