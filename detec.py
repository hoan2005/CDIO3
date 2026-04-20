import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class PhatHienKhongMu:
    def __init__(self, model_path=None, conf=0.5, iou=0.45):
        self.conf = conf
        self.iou  = iou
 
        model_path = model_path or CONFIG["model_path"]
        print(f"[INFO] Đang tải model: {model_path}")
        self.model = YOLO(model_path)
 
        self.class_names = self.model.names 
        self.is_custom_model = self._detect_model_type()
 
        self.total_frames    = 0
        self.total_persons   = 0
        self.total_violations = 0
 
        os.makedirs(CONFIG["violation_save_dir"], exist_ok=True)
 
        print(f"[INFO] Model loaded. Chế độ: {'Custom (helmet)' if self.is_custom_model else 'COCO (general)'}")
        print(f"[INFO] Classes: {self.class_names}")
 
    def _detect_model_type(self):
        """Kiểm tra model có phải custom helmet detection không."""
        names = set(self.class_names.values())
        helmet_keywords = {"helmet", "without_helmet", "with_helmet", "no_helmet",
                          "helmet_on", "helmet_off", "rider", "motorcycle_helmet"}
        return bool(names & helmet_keywords)
 
    def process_frame(self, frame):
        self.total_frames += 1
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        violations = []
 
        if self.is_custom_model:
            frame, violations = self._process_custom(frame, results)
        else:
            frame, violations = self._process_coco(frame, results)
 
        frame = self._draw_hud(frame, violations)
        return frame, violations
 
    def _process_custom(self, frame, results):
        violations = []
        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
 
                is_violation = cls_name in ("without_helmet", "no_helmet", "helmet_off")
                color = CONFIG["colors"]["violation_box"] if is_violation else CONFIG["colors"]["safe_box"]
                label = f"{'VI PHAM' if is_violation else 'AN TOAN'} {conf:.2f}"
 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                self._draw_label(frame, label, (x1, y1), color)
 
                if is_violation:
                    violations.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "class": cls_name,
                    })
                    self.total_violations += 1
 
        self.total_persons += len(violations)
        return frame, violations
 
    def _process_coco(self, frame, results):
        violations = []
        persons    = []
        motorcycles = []
 
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                bbox   = tuple(map(int, box.xyxy[0]))
                name   = self.class_names.get(cls_id, "")
 
                if name == "person":
                    persons.append({"bbox": bbox, "conf": conf})
                elif name in ("motorcycle", "bicycle"):
                    motorcycles.append({"bbox": bbox, "conf": conf})
 
        riders = self._find_riders(persons, motorcycles)
 
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            is_rider = p in riders
 
            head_h    = (y2 - y1) // 5
            head_roi  = frame[y1:y1+head_h, x1:x2]
 
            has_helmet = self._heuristic_helmet_check(head_roi) if head_roi.size > 0 else True
 
            is_violation = is_rider and not has_helmet
            color = CONFIG["colors"]["violation_box"] if is_violation else CONFIG["colors"]["person"]
 
            label_text = ""
            if is_violation:
                label_text = f"⚠ VI PHAM {p['conf']:.2f}"
                violations.append({"bbox": p["bbox"], "conf": p["conf"], "class": "no_helmet_heuristic"})
                self.total_violations += 1
            elif is_rider:
                label_text = f"Rider {p['conf']:.2f}"
            else:
                label_text = f"Person {p['conf']:.2f}"
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._draw_label(frame, label_text, (x1, y1), color)
 
            # Vẽ vùng đầu (debug)
            cv2.rectangle(frame, (x1, y1), (x2, y1+head_h),
                         (0, 255, 255) if has_helmet else (0, 0, 255), 1)
 
        for m in motorcycles:
            x1, y1, x2, y2 = m["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), CONFIG["colors"]["rider"], 1)
            self._draw_label(frame, f"Xe may {m['conf']:.2f}", (x1, y1), CONFIG["colors"]["rider"])
 
        self.total_persons += len(persons)
        return frame, violations
 
    def _find_riders(self, persons, motorcycles, overlap_thresh=0.15):
        riders = []
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            for m in motorcycles:
                mx1, my1, mx2, my2 = m["bbox"]
                # Tính overlap
                ix1 = max(px1, mx1); iy1 = max(py1, my1)
                ix2 = min(px2, mx2); iy2 = min(py2, my2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    p_area = (px2-px1) * (py2-py1)
                    if inter / p_area > overlap_thresh:
                        riders.append(p)
                        break
                # Kiểm tra người nằm ngay phía trên xe máy
                if py2 >= my1 - 20 and px1 < mx2 and px2 > mx1:
                    riders.append(p)
                    break
        return riders
 
    def _heuristic_helmet_check(self, head_roi):
        if head_roi.size == 0:
            return True
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 50, 80), (25, 170, 255))
        skin_ratio = np.sum(skin_mask > 0) / head_roi.size
        return skin_ratio < 0.35
 
 
    def _draw_label(self, frame, text, pos, color):
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(y - 5, th + 5)
        cv2.rectangle(frame, (x, y-th-4), (x+tw+4, y+2), color, -1)
        cv2.putText(frame, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
 
    def _draw_hud(self, frame, violations):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
 
        lines = [
            f"Frame: {self.total_frames}",
            f"Vi pham hien tai: {len(violations)}",
            f"Tong vi pham: {self.total_violations}",
        ]
        if CONFIG["show_fps"] and hasattr(self, "_fps"):
            lines.append(f"FPS: {self._fps:.1f}")
 
        for i, line in enumerate(lines):
            color = (0, 0, 255) if ("Vi pham" in line and len(violations) > 0) else (255, 255, 255)
            cv2.putText(frame, line, (10, 25 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
 
        # Cảnh báo lớn khi có vi phạm
        if violations:
            cv2.putText(frame, "! VI PHAM MU BAO HIEM !", (w//2-220, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
 
        return frame
 
    def save_violation(self, frame, violations, frame_id):
        if not violations:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CONFIG["violation_save_dir"], f"violation_{ts}_f{frame_id}.jpg")
        cv2.imwrite(path, frame)
        print(f"[SAVE] Vi phạm đã lưu: {path}")
 