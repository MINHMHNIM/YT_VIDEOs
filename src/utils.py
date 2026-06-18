import os
 
def clean_up_temp_file(file_path):
    # BUG 4 ĐÃ SỬA: file utils.py bị thiếu hoàn toàn, main.py import nhưng không có
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"   [Dọn] Đã xóa file tạm: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"   [Cảnh báo] Không xóa được file tạm: {e}")