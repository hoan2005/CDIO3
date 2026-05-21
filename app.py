import os, json, base64, time, threading, uuid
from flask import Flask, render_template, request, send_from_directory, Response, jsonify
from datetime import timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER  = 'results'
for d in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(d, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── Models ───────────────────────────────────────────────────────────────────
model_traffic = YOLO("yolov8s.pt")
# Nếu chưa có model biển số, dùng coco (class 2=car làm placeholder)
try:
    model_helmet  = YOLO(r"runs/detect/train5/weights/best.pt")
except Exception:
    model_helmet  = None

# ─── Job store (in-memory) ─────────────────────────────────────────────────────
jobs = {}   # job_id -> { status, progress, violations, frames_done, frames_total }

# ─── Geometry helpers ─────────────────────────────────────────────────────────

def check_side(point, line_pts):
    x, y = point
    (x1, y1), (x2, y2) = line_pts
    return (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)

def iou(b1, b2):
    ix1, iy1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    ix2, iy2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter) > 0 else 0

def calculate_ioa(box_person, box_motorbike):
    x_left  = max(box_person[0], box_motorbike[0])
    y_top   = max(box_person[1], box_motorbike[1])
    x_right = min(box_person[2], box_motorbike[2])
    y_bot   = min(box_person[3], box_motorbike[3])
    inter   = max(0, x_right-x_left)*max(0, y_bot-y_top)
    area    = (box_person[2]-box_person[0])*(box_person[3]-box_person[1])
    return inter/area if area > 0 else 0

def get_combined_box(b1, b2):
    return [min(b1[0],b2[0]), min(b1[1],b2[1]), max(b1[2],b2[2]), max(b1[3],b2[3])]

def detect_light_color(roi, k=4):
    try:
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1,3)
        pixels = pixels[pixels[:,2] > 100]
        if len(pixels) < k:
            return "UNKNOWN"
        km     = KMeans(n_clusters=k, n_init=7)
        km.fit(pixels)
        centers = np.array(km.cluster_centers_)
        best    = centers[np.argmax(centers[:,1]*centers[:,2])]
        h = best[0]
        if h < 10 or h > 170: return "RED"
        if 50 < h < 100:       return "GREEN"
        if 15 < h < 35:        return "YELLOW"
    except Exception:
        pass
    return "UNKNOWN"

def get_camera_config(video_name):
    if not os.path.exists('config.json'):
        return None
    with open('config.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)
    return configs.get(video_name)

def ts(seconds):
    return str(timedelta(seconds=int(seconds)))

def crop_b64(frame, box):
    x1,y1,x2,y2 = map(int, box)
    roi = frame[max(0,y1):y2, max(0,x1):x2]
    if roi.size == 0:
        return ""
    _, buf = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()
# hàm lấy biển số
def get_hinhanh(frame,x1,y1,x2,y2):
    plate_img_b64 = crop_b64(frame, [x1,y1,x2,y2]) 
    return plate_img_b64   
# ─── Processing worker ────────────────────────────────────────────────────────

def process_video(job_id, input_path, video_name):
    job = jobs[job_id]
    job['status'] = 'processing'

    config = get_camera_config(video_name)
    if not config:
        # Default config nếu chưa có config.json
        config = {
            "stop_line":        [[0, 400], [1280, 400]],
            "traffic_light_box":[100, 50, 200, 150],
            "location":         "Không rõ địa điểm"
        }

    line_pts   = config['stop_line']
    light_box  = config['traffic_light_box']
    location   = config.get('location', 'Unknown')

    cap     = cv2.VideoCapture(input_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    job['frames_total'] = total

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(orig_h)
    print(orig_w)
    # Nếu video quay dọc (h > w) → cần xoay, width/height hoán đổi cho VideoWriter
    ROTATE = orig_h < orig_w
    print(ROTATE)
    out_w, out_h = (orig_h, orig_w) if ROTATE else (orig_w, orig_h)
 
    out_path = os.path.join(RESULT_FOLDER, f"result_{job_id}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*'avc1')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    history          = {}
    violations       = []
    count_his        = set()
    helmet_vio_ids   = set()
    traffic_color    = "UNKNOWN"
    frame_count      = 0

    # Track thời gian vi phạm để tính from→to
    vio_start_time   = {}  # track_id -> start_second

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if ROTATE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_count += 1
        cur_sec= frame_count / fps
        job['frames_done'] = frame_count
        job['progress']    = int(frame_count / max(total,1) * 100)

        results = model_traffic.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        box_nguoi, box_xe = [], []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box in boxes:
                track_id = int(box.id[0])
                cls      = int(box.cls[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx,cy    = (x1+x2)//2, (y1+y2)//2
                if frame_count % 5 == 0:
                # ── Đèn giao thông ──
                    if  cls == 9:
                        if iou((x1,y1,x2,y2), light_box) > 0.5:
                            traffic_color = detect_light_color(frame[y1:y2, x1:x2])

                    # ── Người ──
                    if cls == 0:
                        box_nguoi.append({"box": [x1,y1,x2,y2], "id": track_id})
                        # cv2.rectangle(frame, (x1,y1), (x2,y2), (50,200,50), 2)
                        # cv2.putText(frame, f"P:{track_id}", (x1, y1-6),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,200,50), 1)

                    # ── Xe (ô tô, mô tô, xe tải, xe buýt) ──
                    if cls in [2, 3, 5, 7]:
                        if cls == 3:
                            box_xe.append({"box": [x1,y1,x2,y2], "id": track_id})
                        if track_id in count_his:
                                    color = (0,0,230) 
                                    label = "VI PHAM DEN" 
                                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                                    cv2.putText(frame, label, (x1, y1+50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    continue                        
                        cur_side = check_side((cx,cy), line_pts)
                        if track_id in history:
                            prev_side = history[track_id]
                            if prev_side * cur_side < 0 and traffic_color == "RED":
                                if track_id not in count_his:
                                    count_his.add(track_id)
                                    vio_start_time[track_id] = cur_sec
                                    color = (0,0,230) 
                                    label = "VI PHAM DEN" 
                                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                                    cv2.putText(frame, label, (x1, y1+50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                    # ── Biển số ──
                                    plate_img_b64 = get_hinhanh(frame,x1,y1-100,x2,y2)
                                    

                                    violations.append({
                                        "id":        track_id,
                                        "time_from": round(cur_sec),
                                        "time_to":   None,
                                        "location":  location,
                                        "type":      "Vượt đèn đỏ",
                                        "plate_b64": plate_img_b64,
                                    })

                        history[track_id] = cur_side

                        # color = (0,0,230) if track_id in count_his else (0,200,0)
                        # label = "VI PHAM DEN" if track_id in count_his else f"V:{track_id}"
                        # cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        # cv2.putText(frame, label, (x1, y1-8),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── Mũ bảo hiểm ──
        if frame_count % 5==0  and model_helmet:
            print("vào MBH")
            for p in box_nguoi:
                for v in box_xe:
                    if v["id"] in helmet_vio_ids:
                        color = (0,0,230) 
                        label = "VI PHAM MBH"
                        cv2.rectangle(frame, (p["box"][0],p["box"][1]), (p["box"][2],p["box"][3]), color, 2)
                        cv2.putText(frame, label, (p["box"][0],p["box"][1]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        continue
                    if calculate_ioa(p["box"], v["box"]) > 0.2:
                        cb = get_combined_box(p["box"], v["box"])
                        roi = frame[max(0,int(cb[1])):int(cb[3]),
                                    max(0,int(cb[0])):int(cb[2])]
                        if roi.size == 0:
                            continue
                        try:
                            bh_res = model_helmet.predict(source=roi, conf=0.6, verbose=False)
                            for r in bh_res:
                                for bx in r.boxes:
                                    if int(bx.cls[0]) == 1:
                                        
                                        color = (0,0,230) 
                                        label = "VI PHAM MBH"
                                        cv2.rectangle(frame, (p["box"][0],p["box"][1]), (p["box"][2],p["box"][3]), color, 2)
                                        cv2.putText(frame, label, (p["box"][0],p["box"][1]+50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)                                        
                                        helmet_vio_ids.add(v["id"])
                                        # ── Biển số ──
                                        plate_img_b64 = get_hinhanh(frame,cb[0],cb[1],cb[2],cb[3])
                                       
                                        violations.append({
                                            "id":        v["id"],
                                            "time_from": round(cur_sec),
                                            "time_to":   None,
                                            "location":  location,
                                            "type":      "Không đội mũ bảo hiểm",
                                            "plate_b64": plate_img_b64,
                                        })
                        except Exception:
                            pass

        # ── Cập nhật time_to cho vi phạm đang diễn ra ──
        for vio in violations:
            if vio["time_to"] is None and track_id == vio["id"]:
                vio["time_to"] = round(cur_sec)

        # ── Overlay ──
        lc = {"RED":(0,0,220),"GREEN":(0,200,0),"YELLOW":(0,200,220)}.get(traffic_color,(180,180,180))
        cv2.line(frame, tuple(line_pts[0]), tuple(line_pts[1]), (0,220,220), 2)
        cv2.rectangle(frame, (18,12), (280,100), (0,0,0), -1)
        cv2.rectangle(frame, (18,12), (280,100), lc, 2)
        cv2.putText(frame, f"Den: {traffic_color}", (24,38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, lc, 2)
        cv2.putText(frame, f"Vi pham: {len(count_his)}", (24,68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,220), 2)
        cv2.putText(frame, location, (24,92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        out.write(frame)

    cap.release()
    out.release()

    # Đảm bảo time_to hợp lệ
    for vio in violations:
        if vio["time_to"] is None:
            vio["time_to"] = vio["time_from"] + 2

    job['violations'] = violations
    job['result_path'] = out_path
    job['status'] = 'done'


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video_file' not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files['video_file']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    job_id    = str(uuid.uuid4())[:8]
    filename  = secure_filename(f.filename)
    in_path   = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    f.save(in_path)

    jobs[job_id] = {
        "status":       "queued",
        "progress":     0,
        "frames_done":  0,
        "frames_total": 0,
        "violations":   [],
        "result_path":  None,
        "orig_name":    filename,
    }

    t = threading.Thread(target=process_video,
                         args=(job_id, in_path, filename), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status":       job["status"],
        "progress":     job["progress"],
        "frames_done":  job["frames_done"],
        "frames_total": job["frames_total"],
    })


@app.route('/result/<job_id>')
def result(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404

    violations = []
    for v in job["violations"]:
        t_from = int(v["time_from"])
        t_to   = int(v["time_to"]) if v["time_to"] else t_from + 2
        violations.append({
            "id":       v["id"],
            "time":     f"{t_from:02d}s – {t_to:02d}s",
            "location": v["location"],
            "type":     v["type"],
            "plate_b64": v.get("plate_b64", ""),
        })

    return jsonify({
        "violations": violations,
        "video_url":  f"/video/{job_id}",
    })


@app.route('/video/<job_id>')
def serve_video(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("result_path"):
        return "Not found", 404
    dirname  = os.path.dirname(job["result_path"])
    basename = os.path.basename(job["result_path"])
    return send_from_directory(os.path.abspath(dirname), basename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
