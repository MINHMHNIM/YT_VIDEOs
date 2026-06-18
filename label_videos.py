"""
Label video bằng Groq Whisper large-v3.
- Miễn phí: 7200 giây audio/giờ, reset mỗi giờ
- Chính xác: Whisper large-v3 (tốt nhất hiện tại)
- Nhanh: chạy trên cloud, không tốn RAM máy

Cài: pip install groq
Thêm .env: GROQ_API_KEY=your_key
Lấy key tại: https://console.groq.com/keys
"""
import os
import csv
import time
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')
FFMPEG_BIN  = os.path.join(FFMPEG_PATH, 'ffmpeg.exe') if FFMPEG_PATH else 'ffmpeg'
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()

if not GROQ_API_KEY:
    raise ValueError("Thiếu GROQ_API_KEY trong .env! Lấy tại: https://console.groq.com/keys")

# ========== CẤU HÌNH ==========
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR   = os.path.join(BASE_DIR, 'data', 'test')
LABEL_DIR   = os.path.join(BASE_DIR, 'data', 'labels')
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'dataset')
TEMP_DIR    = os.path.join(BASE_DIR, 'data', 'temp_audio')

# Groq free tier: 7200 giây audio/giờ — delay nhẹ là đủ
DELAY_BETWEEN_REQUESTS = 3


def extract_audio_wav(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    subprocess.run([
        FFMPEG_BIN, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_with_groq(audio_path):
    """Groq Whisper large-v3 với word-level timestamps"""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_path, 'rb') as f:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f),
            model="whisper-large-v3",
            language="vi",
            response_format="verbose_json",
            timestamp_granularities=["word"],  # word-level timestamps
        )

    words = []
    if hasattr(response, 'words') and response.words:
        for w in response.words:
            word  = w['word']  if isinstance(w, dict) else w.word
            start = w['start'] if isinstance(w, dict) else w.start
            end   = w['end']   if isinstance(w, dict) else w.end
            words.append({
                "word":  str(word).strip(),
                "start": round(float(start), 2),
                "end":   round(float(end), 2),
            })

    full_text = response.text if hasattr(response, 'text') else ""
    return words, full_text


def save_label_csv(words, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Start', 'End'])
        for w in words:
            writer.writerow([w['word'], w['start'], w['end']])


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    videos = sorted([f for f in os.listdir(FINAL_DIR) if f.endswith('.mp4')])
    existing = set(f.replace('.csv', '.mp4') for f in os.listdir(LABEL_DIR)
                   if f.endswith('.csv'))
    to_process = [v for v in videos if v not in existing]

    print(f"=== LABEL VIDEO BẰNG GROQ WHISPER ===")
    print(f"Model: whisper-large-v3")
    print(f"Tổng video: {len(videos)}")
    print(f"Đã có label: {len(existing)}")
    print(f"Cần label: {len(to_process)}\n")

    if not to_process:
        print("Tất cả video đã có label!")
        return

    success = 0
    errors  = 0

    for i, video_name in enumerate(to_process):
        video_path = os.path.join(FINAL_DIR, video_name)
        csv_name   = video_name.replace('.mp4', '.csv')
        csv_path   = os.path.join(LABEL_DIR, csv_name)
        audio_path = os.path.join(TEMP_DIR, video_name.replace('.mp4', '.wav'))

        print(f"[{i+1}/{len(to_process)}] {video_name}")
        extract_audio_wav(video_path, audio_path)

        try:
            words, full_text = transcribe_with_groq(audio_path)
        except Exception as e:
            msg = str(e)
            if 'rate' in msg.lower() or '429' in msg:
                print(f"   [Rate limit] Chờ 60s...")
                time.sleep(60)
                try:
                    words, full_text = transcribe_with_groq(audio_path)
                except Exception as e2:
                    print(f"   [Lỗi] {str(e2)[:100]}")
                    errors += 1
                    continue
            else:
                print(f"   [Lỗi] {msg[:100]}")
                errors += 1
                continue

        if words:
            save_label_csv(words, csv_path)
            success += 1
            shutil.copy2(video_path, os.path.join(DATASET_DIR, video_name))
            shutil.copy2(csv_path,   os.path.join(DATASET_DIR, csv_name))
            preview = full_text[:80] + ('...' if len(full_text) > 80 else '')
            print(f"   ✓ {len(words)} words | {preview}")
        else:
            print(f"   ✗ Không tìm thấy từ nào")
            errors += 1

        if os.path.exists(audio_path):
            os.remove(audio_path)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n=== HOÀN TẤT ===")
    print(f"Thành công: {success}/{len(to_process)}")
    if errors:
        print(f"Lỗi: {errors} (chạy lại sẽ tự thử lại)")
    print(f"Dataset: {DATASET_DIR}")


if __name__ == "__main__":
    main()