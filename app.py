from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import json
import numpy as np
from datetime import timedelta
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sklearn.cluster import KMeans

app = Flask(__name__)

# Cấu hình thư mục
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải model YOLOv8
model = YOLO("yolov8s.pt")
model_bh=YOLO(r"runs\detect\train5\weights\best.pt")
def calculate_ioa(box_person, box_motorbike):
    """
    Tính toán IoA: Diện tích giao nhau chia cho diện tích của Box Người.
    box_person: [x1, y1, x2, y2]
    box_motorbike: [x1, y1, x2, y2]
    """
    # 1. Xác định tọa độ của vùng giao nhau (Intersection)
    x_left = max(box_person[0], box_motorbike[0])
    y_top = max(box_person[1], box_motorbike[1])
    x_right = min(box_person[2], box_motorbike[2])
    y_bottom = min(box_person[3], box_motorbike[3])

    # 2. Tính diện tích vùng giao nhau
    # Nếu x_right < x_left hoặc y_bottom < y_top thì diện tích giao nhau bằng 0
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # 3. Tính diện tích của Box Người (Area of Person)
    person_area = (box_person[2] - box_person[0]) * (box_person[3] - box_person[1])

    # 4. Tránh lỗi chia cho 0 nếu box người không hợp lệ
    if person_area <= 0:
        return 0

    # 5. Tính tỷ lệ IoA
    ioa = intersection_area / person_area
    
    return ioa
def get_combined_box(box_person, box_motorbike):
    """
    Tạo một box mới bao quanh cả người và xe máy.
    Định dạng box đầu vào: [x1, y1, x2, y2]
    """
    # Lấy giá trị nhỏ nhất của x1 và y1 (góc trên bên trái)
    x1_new = min(box_person[0], box_motorbike[0])
    y1_new = min(box_person[1], box_motorbike[1])
    
    # Lấy giá trị lớn nhất của x2 và y2 (góc dưới bên phải)
    x2_new = max(box_person[2], box_motorbike[2])
    y2_new = max(box_person[3], box_motorbike[3])
    
    return [x1_new, y1_new, x2_new, y2_new]
# --- HÀM LOGIC HÌNH HỌC (f(x,y) = (x2-x1)(y-y1) - (y2-y1)(x-x1)) ---
def check_side(point, line_pts):
    x, y = point
    (x1, y1), (x2, y2) = line_pts
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def detect_traffic_light_color_kmeans(roi, k=4):
    try:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixels = hsv_roi.reshape(-1, 3)
        pixels = pixels[pixels[:, 2] > 100]
        if len(pixels) == 0: return "UNKNOWN"
        kmeans = KMeans(n_clusters=k, n_init=7)
        kmeans.fit(pixels)
        centers = np.array(kmeans.cluster_centers_)
        scores = centers[:, 1] * centers[:, 2]
        best = centers[np.argmax(scores)]
        h = best[0]
        if (h < 10) or (h > 170): return "RED"
        elif 50 < h < 100: return "GREEN"
        elif 15 < h < 35: return "YELLOW"
    except: pass
    return "UNKNOWN"

def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_camera_config(video_name):
    if not os.path.exists('config.json'): return None
    with open('config.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)
    return configs.get(video_name)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_process():
    if 'video_file' not in request.files: return "No file part"
    file = request.files['video_file']
    if file.filename == '': return "No selected file"

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = "processed_" + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    config = get_camera_config(filename)
    if not config:
        return f"Thiếu cấu hình cho file {filename} trong config.json"

    line_pts = config['stop_line'] 
    light_box = config['traffic_light_box']
    location_name = config.get('location', 'Unknown Road')

    cap = cv2.VideoCapture(input_path)
    # Lấy thông số gốc
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # NẾU CÓ XOAY 90 ĐỘ: Width và Height sẽ hoán đổi cho nhau
    # Ở đây tôi giả định bạn cần xoay như trong code bạn gửi
    out_w, out_h = orig_h, orig_w 
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    history = {} 
    violation_list = [] 
    count_his = set()
    traffic_light_color = "UNKNOWN"
    frame_count = 0
    helmet_violation_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Xoay frame theo yêu cầu của bạn
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        timestamp = str(timedelta(seconds=int(frame_count/fps)))

        # Chạy Tracking
        results = model.track(frame, persist=True, verbose=False,tracker="custom_bytetrack.yaml")
        box_nguoi=[]
        box_xe=[]
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box in boxes:
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 1. Kiểm tra đèn
                if frame_count%10==0:
                  if cls == 9: 
                    if iou((x1, y1, x2, y2), light_box) > 0.5:
                        traffic_light_color = detect_traffic_light_color_kmeans(frame[y1:y2, x1:x2])
                        cv2.putText(frame, traffic_light_color, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if cls ==0:
                    box_nguoi.append({"box":[x1, y1, x2, y2],"id":track_id})
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                # 2. Kiểm tra vi phạm
                if cls in [2, 3, 5, 7]:
                    if cls ==3:
                        box_xe.append({"box":[x1, y1, x2, y2],"id":track_id})
                    current_side = check_side((cx, cy), line_pts)
                    if track_id in history:
                        prev_side = history[track_id]
                        # Logic đổi dấu phương trình đường thẳng
                        if prev_side * current_side < 0 and traffic_light_color == "RED":
                            if track_id not in count_his:
                                count_his.add(track_id)
                                violation_list.append({
                                    "id": track_id,
                                    "time": timestamp,
                                    "location": location_name,
                                    "type": "Vượt đèn đỏ"
                                })
                    
                    history[track_id] = current_side
                    
                    # Vẽ Box và ID
                    color = (0, 0, 255) if track_id in count_his else (0, 255, 0)
                    label = "VI PHAM DEN" if track_id in count_his else f"ID:{track_id}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if frame_count % 5 == 0:
            for p in box_nguoi:
                for v in box_xe:
                    ioa = calculate_ioa(p["box"], v["box"])
                    print(ioa)
                    # Nếu người đang ngồi trên xe (IoA > 0.5)
                    if ioa > 0.2:
                        # Tránh kiểm tra lại nếu xe này đã bị ghi nhận vi phạm mũ bảo hiểm
                        if v["id"] in helmet_violation_ids:
                            continue
                        
                        x1_new, y1_new, x2_new, y2_new = get_combined_box(p["box"], v["box"])
                        # Đảm bảo tọa độ không âm và nằm trong frame
                        roi_combined = frame[max(0, int(y1_new)):int(y2_new), max(0, int(x1_new)):int(x2_new)]
                        
                        if roi_combined.size == 0: continue

                        # Dự đoán mũ bảo hiểm
                        results_bh = model_bh.predict(source=roi_combined, save=False, conf=0.6, verbose=False)
                        
                        # Logic: Giả sử model_bh trả về các box là "mũ bảo hiểm"
                        # Nếu KHÔNG tìm thấy box nào trong vùng ROI này thì coi như vi phạm
                        for result in results_bh:
                            boxes = result.boxes  # Các bounding box tìm thấy
                            for box in boxes:
                                # Tọa độ [x1, y1, x2, y2]
                                cls = int(box.cls[0])
                                print(cls)
                                x1,y1,x2,y2 = box.xyxy[0].tolist()
                                if cls==1:
                                    helmet_violation_ids.add(v["id"])
                                    violation_list.append({
                                        "id": v["id"],
                                        "time": timestamp,
                                        "location": location_name,
                                        "type": "Không đội mũ bảo hiểm"
                                    })


        # Vẽ Overlay tĩnh
        cv2.line(frame, tuple(line_pts[0]), tuple(line_pts[1]), (255, 255, 0), 3)
        cv2.putText(frame, f"Loi: {len(count_his)}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, location_name, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame)

    cap.release()
    out.release()

    # Trả về trang HTML cùng với dữ liệu vi phạm
    return render_template('index.html', 
                            original=filename, 
                            processed=output_filename, 
                            violations=violation_list)

@app.route('/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)