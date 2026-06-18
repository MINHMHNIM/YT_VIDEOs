"""
Chia video từ final/ thành train/test theo thời gian lưu về.
- Video tải về sớm nhất → train
- Video tải về sau → test
- Mốc: 5 giờ đầu vào train, còn lại vào test
"""
import os
import shutil
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR  = os.path.join(BASE_DIR, 'data', 'final')
TRAIN_DIR  = os.path.join(BASE_DIR, 'data', 'train')
TEST_DIR   = os.path.join(BASE_DIR, 'data', 'test')

TRAIN_HOURS = 5


def get_duration(video_path):
    ffprobe = os.path.join(FFMPEG_PATH, 'ffprobe.exe') if FFMPEG_PATH else 'ffprobe'
    result = subprocess.run(
        [ffprobe, '-v', 'quiet', '-print_format', 'json', '-show_format', video_path],
        capture_output=True, text=True
    )
    try:
        return float(json.loads(result.stdout)['format']['duration'])
    except:
        return 0


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Lấy tất cả video + thời gian tạo file
    videos = []
    for f in os.listdir(FINAL_DIR):
        if not f.endswith('.mp4'):
            continue
        path = os.path.join(FINAL_DIR, f)
        ctime = os.path.getmtime(path)  # thời gian lưu về
        videos.append((f, path, ctime))

    # Sắp xếp theo thời gian lưu — cũ nhất trước
    videos.sort(key=lambda x: x[2])

    print(f"=== CHIA DATASET THEO THỜI GIAN LƯU ===")
    print(f"Tổng video: {len(videos)}")
    print(f"Train: {TRAIN_HOURS} giờ đầu tiên lưu về")
    print(f"Test: phần còn lại\n")

    import datetime
    train_sec = 0
    train_count = 0
    test_count = 0
    target_sec = TRAIN_HOURS * 3600

    for video_name, video_path, ctime in videos:
        duration = get_duration(video_path)
        time_str = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')

        if train_sec < target_sec:
            shutil.copy2(video_path, os.path.join(TRAIN_DIR, video_name))
            train_sec += duration
            train_count += 1
            label = "TRAIN"
        else:
            shutil.copy2(video_path, os.path.join(TEST_DIR, video_name))
            test_count += 1
            label = "TEST"

        print(f"  [{label}] {video_name} | lưu lúc {time_str} | {duration:.1f}s")

    test_sec = sum(get_duration(os.path.join(TEST_DIR, f))
                   for f in os.listdir(TEST_DIR) if f.endswith('.mp4'))

    print(f"\n=== KẾT QUẢ ===")
    print(f"Train: {train_count} video ({train_sec/3600:.2f} giờ) → {TRAIN_DIR}")
    print(f"Test:  {test_count} video ({test_sec/3600:.2f} giờ) → {TEST_DIR}")
    print(f"Tổng: {(train_sec+test_sec)/3600:.2f} giờ")


if __name__ == "__main__":
    main()