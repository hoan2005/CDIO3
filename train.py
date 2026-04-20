from ultralytics import YOLO
import yaml
import os

TRAIN_CONFIG = {
    "base_model": "yolov8n.pt",     
    # "base_model": "yolov8s.pt",   
    # "base_model": "yolov8m.pt",   

    "data_yaml":    "dataset/data.yaml",
    "epochs":       100,
    "batch_size":   16,
    "image_size":   640,
    "workers":      4,
    "device":       "0",     
    "project":      "runs/helmet_detection",
    "name":         "exp",
    "patience":     20,     
    "save_period":  10,      
}

SAMPLE_DATA_YAML = {
    "path": "../dataset",
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",   

    "nc": 2,    
    "names": {
        0: "with_helmet",
        1: "without_helmet",
        # 2: "rider",       
    },
}


def create_sample_yaml(output_path="dataset/data.yaml"):
    """Tạo file data.yaml mẫu."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(SAMPLE_DATA_YAML, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] Đã tạo data.yaml mẫu: {output_path}")
    print("[INFO] Hãy cập nhật đường dẫn và đặt ảnh vào dataset/train/ và dataset/val/")

def train():
    if not os.path.exists(TRAIN_CONFIG["data_yaml"]):
        print("[WARN] Chưa có data.yaml. Đang tạo file mẫu...")
        create_sample_yaml(TRAIN_CONFIG["data_yaml"])
        print("\n[STOP] Vui lòng:")
        print("  1. Đặt ảnh vào dataset/train/images/ và dataset/val/images/")
        print("  2. Đặt label vào dataset/train/labels/ và dataset/val/labels/")
        print("  3. Cập nhật path trong dataset/data.yaml")
        print("  4. Chạy lại script này\n")
        return

    print(f"[INFO] Bắt đầu training với model: {TRAIN_CONFIG['base_model']}")
    print(f"[INFO] Epochs: {TRAIN_CONFIG['epochs']}, Batch: {TRAIN_CONFIG['batch_size']}")

    model = YOLO(TRAIN_CONFIG["base_model"])

    results = model.train(
        data      = TRAIN_CONFIG["data_yaml"],
        epochs    = TRAIN_CONFIG["epochs"],
        batch     = TRAIN_CONFIG["batch_size"],
        imgsz     = TRAIN_CONFIG["image_size"],
        workers   = TRAIN_CONFIG["workers"],
        device    = TRAIN_CONFIG["device"],
        project   = TRAIN_CONFIG["project"],
        name      = TRAIN_CONFIG["name"],
        patience  = TRAIN_CONFIG["patience"],
        save_period = TRAIN_CONFIG["save_period"],
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 5.0,
        flipud    = 0.1,
        mosaic    = 1.0,
        mixup     = 0.1,
        optimizer = "AdamW",
        lr0       = 0.001,
        cos_lr    = True,
    )

    best_model_path = f"{TRAIN_CONFIG['project']}/{TRAIN_CONFIG['name']}/weights/best.pt"
    print(f"\n[DONE] Training hoàn tất!")
    print(f"[DONE] Best model: {best_model_path}")

    print("\n[INFO] Đang validate model tốt nhất...")
    best_model = YOLO(best_model_path)
    metrics    = best_model.val(data=TRAIN_CONFIG["data_yaml"])
    print(f"[RESULT] mAP50: {metrics.box.map50:.4f}")
    print(f"[RESULT] mAP50-95: {metrics.box.map:.4f}")

    return best_model_path

def export_model(model_path, format="onnx"):
    """
    Export model sang định dạng khác để deploy.
    Formats: onnx, torchscript, tflite, coreml, engine (TensorRT)
    """
    model = YOLO(model_path)
    exported = model.export(format=format)
    print(f"[EXPORT] Model đã export sang {format}: {exported}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO helmet detection model")
    parser.add_argument("--create-yaml", action="store_true", help="Chỉ tạo data.yaml mẫu")
    parser.add_argument("--export", default=None, help="Export model: onnx/tflite/engine")
    parser.add_argument("--model",  default=None, help="Model path để export")
    args = parser.parse_args()

    if args.create_yaml:
        create_sample_yaml()
    elif args.export and args.model:
        export_model(args.model, args.export)
    else:
        train()