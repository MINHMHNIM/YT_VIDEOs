"""
Chạy 1 lần để tải DNN Face Detector model vào thư mục src/
Chính xác hơn Haar Cascade rất nhiều.
"""
import urllib.request
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

files = {
    'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
}

for name, url in files.items():
    path = os.path.join(SRC_DIR, name)
    if os.path.exists(path):
        print(f"✓ {name} đã có")
        continue
    print(f"Đang tải {name}...")
    urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"✓ Đã tải {name} ({size_mb:.1f}MB)")

print("\nHoàn tất! DNN Face Detector đã sẵn sàng.")
