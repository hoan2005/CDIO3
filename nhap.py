# import cv2
# import os

# # 1. Cấu hình đường dẫn
# input_folder = r'D:\cdio3\so'
# output_folder = r'D:\cdio3\ouput'

# # Tạo thư mục đầu ra nếu chưa có
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 2. Duyệt qua tất cả các file trong thư mục
# for video_name in os.listdir(input_folder):
#     # Kiểm tra định dạng file (có thể thêm .avi, .mkv nếu cần)
#     if video_name.endswith((".mp4", ".mov", ".avi")):
#         video_path = os.path.join(input_folder, video_name)
        
#         # Tạo thư mục riêng cho từng video để tránh ghi đè ảnh
#         video_output_path = os.path.join(output_folder, os.path.splitext(video_name)[0])
#         if not os.path.exists(video_output_path):
#             os.makedirs(video_output_path)

#         # 3. Đọc video
#         cap = cv2.VideoCapture(video_path)
#         frame_count = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break # Hết video hoặc lỗi
            
#             # (Tùy chọn) Xoay 90 độ nếu bạn muốn
#             # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#             # rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#             # 4. Lưu ảnh
#             img_name = f"frame_{frame_count:05d}.jpg"
#             img_path = os.path.join(video_output_path, img_name)
#             cv2.imwrite(img_path, frame)
            
#             frame_count += 1

#         cap.release()
#         print(f"Hoàn thành: {video_name} -> {frame_count} ảnh.")

# print("--- TẤT CẢ ĐÃ XỬ LÝ XONG ---")


import shutil
import os
import uuid
from pathlib import Path

def copy_yolo_mixed_folder(source_dir, img_target, txt_target):
    # Tạo thư mục đích nếu chưa có
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(txt_target, exist_ok=True)

    # Chuyển đường dẫn về dạng Path để xử lý cho mượt
    src_path = Path(source_dir)
    
    # Lấy danh sách file ảnh
    img_files = list(src_path.glob("*.jpg"))
    
    if not img_files:
        print("Không tìm thấy file .jpg nào trong thư mục nguồn!")
        return

    count = 0
    for img_file in img_files:
        # Lấy tên file gốc (không đuôi)
        file_stem = img_file.stem
        
        # Xác định file .txt tương ứng cùng thư mục
        txt_file = src_path / f"{file_stem}.txt"
        
        if txt_file.exists():
            # Tạo tên ngẫu nhiên chung cho cả 2
            random_name = uuid.uuid4().hex
            
            # Copy ảnh sang thư mục ảnh mới
            shutil.copy2(img_file, Path(img_target) / f"{random_name}.jpg")
            
            # Copy nhãn sang thư mục nhãn mới
            shutil.copy2(txt_file, Path(txt_target) / f"{random_name}.txt")
            
            count += 1
            print(f"✅ Đã khớp: {file_stem} -> {random_name}")
        else:
            print(f"⚠️ Bỏ qua: {img_file.name} (Thiếu file .txt tương ứng)")

    print(f"\nHoàn thành! Đã copy {count} cặp file sang 2 thư mục riêng biệt.")

# --- CẤU HÌNH ---
SOURCE = r"D:\cdio3\ouput\video38" # Thư mục chứa chung cả jpg và txt
IMG_DEST = r"D:\cdio3\data\image"             # Thư mục chỉ lưu ảnh sau khi đổi tên
TXT_DEST = r"D:\cdio3\data\label"             # Thư mục chỉ lưu nhãn sau khi đổi tên

copy_yolo_mixed_folder(SOURCE, IMG_DEST, TXT_DEST)