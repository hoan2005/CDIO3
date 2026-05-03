import cv2
import os

# 1. Cấu hình đường dẫn
input_folder = r'D:\cdio3\videobaohiem'
output_folder = r'D:\cdio3\ouput'

# Tạo thư mục đầu ra nếu chưa có
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Duyệt qua tất cả các file trong thư mục
for video_name in os.listdir(input_folder):
    # Kiểm tra định dạng file (có thể thêm .avi, .mkv nếu cần)
    if video_name.endswith((".mp4", ".mov", ".avi")):
        video_path = os.path.join(input_folder, video_name)
        
        # Tạo thư mục riêng cho từng video để tránh ghi đè ảnh
        video_output_path = os.path.join(output_folder, os.path.splitext(video_name)[0])
        if not os.path.exists(video_output_path):
            os.makedirs(video_output_path)

        # 3. Đọc video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break # Hết video hoặc lỗi
            
            # (Tùy chọn) Xoay 90 độ nếu bạn muốn
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # 4. Lưu ảnh
            img_name = f"frame_{frame_count:05d}.jpg"
            img_path = os.path.join(video_output_path, img_name)
            cv2.imwrite(img_path, frame)
            
            frame_count += 1

        cap.release()
        print(f"Hoàn thành: {video_name} -> {frame_count} ảnh.")

print("--- TẤT CẢ ĐÃ XỬ LÝ XONG ---")