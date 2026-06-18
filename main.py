import os
import re
import random
import time
from src import yt_download, audio, visual, utils

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
RAW_DIR   = os.path.join(DATA_DIR, 'raw_vid')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
FINAL_DIR = os.path.join(DATA_DIR, 'final')
PROCESSED_FILE = os.path.join(BASE_DIR, 'processed.txt')


def load_processed():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_processed(vid_id):
    with open(PROCESSED_FILE, 'a') as f:
        f.write(vid_id + '\n')


def get_total_duration_hours(folder):
    import subprocess, json
    total_sec = 0
    if not os.path.exists(folder):
        return 0
    for f in os.listdir(folder):
        if not f.endswith('.mp4'):
            continue
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', os.path.join(folder, f)],
            capture_output=True, text=True
        )
        try:
            info = json.loads(result.stdout)
            total_sec += float(info['format']['duration'])
        except:
            pass
    return total_sec / 3600


def _process_video(vid_id, hq_video_path, accumulated_hours, target_hours):
    """Xử lý 1 video đã tải: tách audio → VAD → cắt clip"""
    print(f"[{vid_id}] 3. Tách âm thanh 16kHz...")
    audio_path = os.path.join(AUDIO_DIR, f"{vid_id}.wav")
    audio.extract_audio(hq_video_path, audio_path)

    print(f"[{vid_id}] 4. Chạy VAD + merge...")
    timestamps = audio.detect_active_speech(audio_path, gap_threshold=5.0)

    print(f"[{vid_id}] 5. Phân tích + cắt clip...")
    saved = 0

    for i, ts in enumerate(timestamps):
        if target_hours and accumulated_hours >= target_hours:
            print(f"   ✓ Đủ giờ, dừng cắt.")
            break

        duration = ts['end'] - ts['start']
        if duration <= 2.0:
            continue

        result = visual.analyze_segment_full(
            hq_video_path, ts['start'], ts['end'],
            min_face_ratio=0.4,
            min_lip_ratio=0.2
        )
        if not result['has_speaker']:
            continue

        real_start = result['real_start']
        real_end   = result['real_end']

        freeze_at = visual.detect_freeze_in_segment(
            hq_video_path, real_start, real_end,
            min_freeze_duration=0.5
        )
        if freeze_at is not None:
            real_end = max(real_start, freeze_at - 0.2)

        real_duration = real_end - real_start
        if real_duration <= 2.0:
            continue

        chunk_name = f"{vid_id}_chunk_{saved:03d}.mp4"
        chunk_path = os.path.join(FINAL_DIR, chunk_name)
        visual.cut_video_segment(hq_video_path, chunk_path, real_start, real_end)
        saved += 1
        accumulated_hours += real_duration / 3600

        trim_s = real_start - ts['start']
        trim_e = ts['end']   - real_end
        print(f"   + {chunk_name} ({real_duration:.1f}s) "
              f"[trim đầu={trim_s:.1f}s đuôi={trim_e:.1f}s] "
              f"| Tổng: {accumulated_hours:.2f}h")

    print(f"   => Giữ {saved}/{len(timestamps)} đoạn")
    return accumulated_hours


def process_single_video(url_or_id_or_path):
    """
    Xử lý 1 video — hỗ trợ 3 cách:
      1. File local:  process_single_video(r"D:\ytvid\raw.mp4")
      2. YouTube URL: process_single_video("https://youtube.com/watch?v=abc123")
      3. Video ID:    process_single_video("abc123")
    """
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    if os.path.isfile(url_or_id_or_path):
        # File local — không cần tải
        hq_path = url_or_id_or_path
        vid_id  = os.path.splitext(os.path.basename(hq_path))[0]
        print(f"=== XỬ LÝ FILE LOCAL: {os.path.basename(hq_path)} ===")
        print(f"1. Quét khuôn mặt + khẩu hình...")
        has_speaker = visual.check_lip_movement(hq_path, max_frames_to_check=900)
        if not has_speaker:
            print(" -> Không phát hiện người nói. Vẫn xử lý tiếp...")
        else:
            print(" -> Phát hiện có người nói!")
    else:
        # YouTube URL hoặc video ID
        match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url_or_id_or_path)
        vid_id = match.group(1) if match else url_or_id_or_path.strip()
        print(f"=== XỬ LÝ VIDEO YOUTUBE: {vid_id} ===")

        print(f"1. Tải bản mờ (144p)...")
        test_path = yt_download.download_worst_video_for_test(vid_id, RAW_DIR)
        if not test_path:
            print("Không tải được video!")
            return

        print(f"2. Quét khuôn mặt + khẩu hình...")
        has_speaker = visual.check_lip_movement(test_path, max_frames_to_check=900)
        utils.clean_up_temp_file(test_path)
        if not has_speaker:
            print(" -> Không phát hiện người nói. Vẫn xử lý tiếp...")

        print(f"3. Tải bản nét (720p)...")
        hq_path = yt_download.download_video_full(vid_id, RAW_DIR)
        if not hq_path:
            print("Không tải được bản full!")
            return

    _process_video(vid_id, hq_path,
                   get_total_duration_hours(FINAL_DIR),
                   target_hours=None)
    print(f"\n=== HOÀN TẤT ===")
    print(f"Clips lưu tại: {FINAL_DIR}")


def run_pipeline(queries=None, target_hours=10):
    print("=== KHỞI ĐỘNG HỆ THỐNG THU THẬP DỮ LIỆU ===")
    print(f"Mục tiêu: {target_hours} giờ clip")

    if queries is None:
        queries = yt_download.QUERIES

    print(f"Số query: {len(queries)}\n")

    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    processed = load_processed()
    print(f"Đã xử lý trước đó: {len(processed)} video")

    accumulated_hours = get_total_duration_hours(FINAL_DIR)
    print(f"Thời lượng hiện có: {accumulated_hours:.2f}h\n")

    if accumulated_hours >= target_hours:
        print(f"✓ Đã đủ {target_hours} giờ ngay từ đầu!")
        return

    for vid_id in yt_download.iter_video_ids_multi_query(queries):

        if accumulated_hours >= target_hours:
            print(f"\n✓ Đã đủ {target_hours} giờ! Dừng lại.")
            break

        if vid_id in processed:
            print(f"[{vid_id}] Đã xử lý, bỏ qua.")
            continue

        print(f"\n{'─'*55}")
        print(f"Tiến độ: {accumulated_hours:.2f}/{target_hours}h "
              f"({accumulated_hours/target_hours*100:.1f}%)")
        print(f"[{vid_id}] 1. Tải bản mờ (144p)...")

        test_video_path = yt_download.download_worst_video_for_test(vid_id, RAW_DIR)
        if not test_video_path:
            save_processed(vid_id)
            continue

        print(f"[{vid_id}] 2. Quét khuôn mặt + khẩu hình...")
        has_speaker = visual.check_lip_movement(test_video_path, max_frames_to_check=900)
        utils.clean_up_temp_file(test_video_path)

        if not has_speaker:
            print(f" -> [BỎ QUA] Không có phát thanh viên.")
            save_processed(vid_id)
            time.sleep(random.uniform(1, 3))
            continue

        print(f" -> [ĐẠT] Tải bản nét (720p)...")
        hq_video_path = yt_download.download_video_full(vid_id, RAW_DIR)
        if not hq_video_path:
            save_processed(vid_id)
            continue

        accumulated_hours = _process_video(
            vid_id, hq_video_path, accumulated_hours, target_hours
        )
        save_processed(vid_id)

        wait = random.uniform(8, 20)
        print(f"   ⏳ Nghỉ {wait:.0f}s trước video tiếp theo...")
        time.sleep(wait)

    print(f"\n=== HOÀN TẤT: {accumulated_hours:.2f} giờ clip ===")


if __name__ == "__main__":
    # ── Xử lý pipeline bình thường ──
    #run_pipeline(
    #    queries=None,
    #   target_hours=6,
    #)

    #process_single_video(r"D:\ytvid\raw1.mp4")
    process_single_video("https://www.youtube.com/watch?v=pQHBB2XBndE&t=763s")