import os
import cv2
import subprocess
from collections import deque
from dotenv import load_dotenv

load_dotenv()
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')
FFMPEG_BIN = os.path.join(FFMPEG_PATH, 'ffmpeg.exe') if FFMPEG_PATH else 'ffmpeg'

# ========== DNN Face Detector ==========
# Dùng OpenCV built-in DNN model 
PROTOTXT = os.path.join(os.path.dirname(__file__), 'deploy.prototxt')
CAFFEMODEL = os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel')
_dnn_net = None

def _get_face_detector():
    global _dnn_net
    if _dnn_net is None:
        if os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL):
            _dnn_net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
            print("   [DNN] Dùng DNN face detector (chính xác cao)")
        else:
            print("   [Haar] DNN model chưa tải, dùng Haar Cascade")
            _dnn_net = "haar"
    return _dnn_net

def detect_faces_dnn(gray, frame_bgr, w, h, confidence_threshold=0.3):
    net = _get_face_detector()
    if net == "haar":
        return detect_faces_haar(gray, w, h)
    
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < confidence_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        fw = x2 - x1
        fh = y2 - y1
        if fw > 0 and fh > 0:
            faces.append((x1, y1, fw, fh, conf))
    return faces

# Haar Cascade backup
_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces_haar(gray, w, h):
    faces = _haar_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    return [(x, y, fw, fh, 1.0) for (x, y, fw, fh) in faces]


def is_anchor_face(x, y, fw, fh, w, h):
    ratio = fw / w
    return ratio >= 0.08


def cut_video_segment(input_video, output_video, start_time, end_time):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    duration = end_time - start_time
    command = [
        FFMPEG_BIN, "-y",
        "-fflags", "+genpts",
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
        "-vf", "fps=25,setpts=PTS-STARTPTS",
        "-af", "aresample=async=1000,asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_video
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def check_lip_movement(video_path, max_frames_to_check=900):
    """Dùng cho video test 144p"""
    cap = cv2.VideoCapture(video_path)
    score = _analyze_segment(cap, max_frames=max_frames_to_check,
                             min_face_size=30)
    cap.release()
    return score['is_talking']


def analyze_segment_full(video_path, start_sec, end_sec,
                          sample_fps=3,
                          min_face_ratio=0.6,
                          min_lip_ratio=0.4):
    """
    ★ HÀM MỚI — 1 lần scan duy nhất, trả về tất cả thông tin:
    - has_speaker: có phát thanh viên đang nói không
    - real_start: giây đầu tiên có anchor nói
    - real_end: giây cuối cùng có anchor nói
    
    Thay thế 3 hàm cũ: check_segment_has_speaker + find_speaker_start + find_speaker_end
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {'has_speaker': False}

    start_frame  = int(start_sec * fps)
    end_frame    = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sample_every = max(1, int(fps / sample_fps))
    
    prev_mouth_roi = None
    diff_history   = deque(maxlen=10)
    
    total_sampled  = 0
    face_frames    = 0
    lip_frames     = 0
    max_faces      = 0
    frame_count    = 0
    
    first_anchor_sec = None
    last_anchor_sec  = None
    first_talking_sec = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_frame = start_frame + frame_count
        frame_count  += 1
        if current_frame > end_frame: break
        if frame_count % sample_every != 0: continue

        total_sampled += 1
        current_sec = current_frame / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect faces
        faces = detect_faces_dnn(gray, frame, w, h)
        anchor_faces = [(x, y, fw, fh) for (x, y, fw, fh, c) in faces
                        if is_anchor_face(x, y, fw, fh, w, h)]

        if len(anchor_faces) == 0:
            prev_mouth_roi = None
            diff_history.clear()
            continue

        max_faces = max(max_faces, len(anchor_faces))
        if len(anchor_faces) > 2:
            prev_mouth_roi = None
            continue

        face_frames += 1
        if first_anchor_sec is None:
            first_anchor_sec = current_sec
        last_anchor_sec = current_sec

        # Lip movement
        x, y, fw, fh = max(anchor_faces, key=lambda f: f[2] * f[3])
        mouth_roi = gray[y + int(fh*0.60):y + int(fh*0.85),
                         x + int(fw*0.20):x + int(fw*0.80)]
        if mouth_roi.size == 0:
            continue
        mouth_roi = cv2.resize(mouth_roi, (30, 15))

        if prev_mouth_roi is not None:
            diff = cv2.absdiff(mouth_roi, prev_mouth_roi)
            diff_history.append(diff.mean())
            if len(diff_history) >= 5:
                if max(diff_history) - min(diff_history) > 2.0 and max(diff_history) > 3.0:
                    lip_frames += 1
                    if first_talking_sec is None:
                        first_talking_sec = current_sec

        prev_mouth_roi = mouth_roi.copy()

    cap.release()

    if total_sampled == 0:
        return {'has_speaker': False}

    face_ratio = face_frames / total_sampled
    lip_ratio  = lip_frames / face_frames if face_frames > 0 else 0

    has_speaker = (
        face_ratio >= min_face_ratio and
        lip_ratio  >= min_lip_ratio  and
        max_faces  <= 2
    )

    # real_start: lùi 0.5s từ lúc bắt đầu nói
    real_start = start_sec
    if first_talking_sec is not None:
        real_start = max(start_sec, first_talking_sec - 0.5)
    elif first_anchor_sec is not None:
        real_start = max(start_sec, first_anchor_sec - 0.5)

    # real_end: thêm 0.5s buffer sau lần cuối thấy anchor
    real_end = end_sec
    if last_anchor_sec is not None:
        real_end = min(end_sec, last_anchor_sec + 0.5)

    return {
        'has_speaker': has_speaker,
        'real_start':  real_start,
        'real_end':    real_end,
        'face_ratio':  round(face_ratio, 2),
        'lip_ratio':   round(lip_ratio, 2),
    }


def detect_freeze_in_segment(video_path, start_sec, end_sec,
                              min_freeze_duration=0.5,
                              noise_threshold=0.001):
    duration = end_sec - start_sec
    command = [
        FFMPEG_BIN, "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"freezedetect=n={noise_threshold}:d={min_freeze_duration}",
        "-an", "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    for line in result.stderr.split("\n"):
        if "freeze_start" in line:
            try:
                freeze_at = float(line.split(":")[-1].strip())
                return start_sec + freeze_at
            except:
                pass
    return None


def _analyze_segment(cap, max_frames, sample_every=3, min_face_size=60):
    """Dùng cho check_lip_movement (video test 144p)"""
    prev_mouth_roi = None
    diff_history   = deque(maxlen=10)
    total_sampled  = 0
    face_frames    = 0
    lip_frames     = 0
    max_faces      = 0
    frame_count    = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if frame_count % sample_every != 0: continue

        total_sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = detect_faces_dnn(gray, frame, w, h)
        anchor_faces = [(x, y, fw, fh) for (x, y, fw, fh, c) in faces
                        if is_anchor_face(x, y, fw, fh, w, h)]

        if len(anchor_faces) == 0:
            prev_mouth_roi = None
            diff_history.clear()
            continue

        max_faces = max(max_faces, len(anchor_faces))
        if len(anchor_faces) > 2:
            prev_mouth_roi = None
            continue

        face_frames += 1
        x, y, fw, fh = max(anchor_faces, key=lambda f: f[2] * f[3])

        mouth_roi = gray[y + int(fh*0.60):y + int(fh*0.85),
                         x + int(fw*0.20):x + int(fw*0.80)]
        if mouth_roi.size == 0: continue
        mouth_roi = cv2.resize(mouth_roi, (30, 15))

        if prev_mouth_roi is not None:
            diff = cv2.absdiff(mouth_roi, prev_mouth_roi)
            diff_history.append(diff.mean())
            if len(diff_history) >= 5:
                if max(diff_history) - min(diff_history) > 2.0 and max(diff_history) > 3.0:
                    lip_frames += 1

        prev_mouth_roi = mouth_roi.copy()

    return {
        'total_sampled': total_sampled,
        'face_frames':   face_frames,
        'lip_frames':    lip_frames,
        'max_faces':     max_faces,
        'is_talking':    face_frames > 0 and lip_frames > 0,
    }