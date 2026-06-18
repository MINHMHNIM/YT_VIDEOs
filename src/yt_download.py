import os
import time
import random
import yt_dlp
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY     = os.getenv('YOUTUBE_API_KEY')
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '')

if not API_KEY:
    raise ValueError("Khong tim thay YOUTUBE_API_KEY. Vui long kiem tra file .env!")

youtube = build('youtube', 'v3', developerKey=API_KEY)

# =====================================================================
QUERIES = [
    "Trực tiếp: Cập nhật diễn biến cơn bão số 3 | VTV24",
    "phóng viên tường thuật trực tiếp cơn bão VTV",
    "phóng viên hiện trường bão lũ VTV24",
    "trực tiếp cập nhật diễn biến bão VTV",
    "bản tin thời sự bão lũ VTV1",
    "phóng sự mưa bão lũ lụt VTV",
]

# =====================================================================
TITLE_MUST_CONTAIN = [
    "bão", "lũ", "mưa", "thời sự", "tin tức", "phóng viên",
    "tường thuật", "hiện trường", "trực tiếp", "cập nhật",
    "bản tin", "thời tiết", "phóng sự",
]

TITLE_MUST_NOT_CONTAIN = [
    "nhạc", "hát", "ca sĩ", "MV", "music", "karaoke", "remix",
    "cover", "live show", "concert", "album", "lyric", "lyrics",
    "phim", "trailer", "review phim", "reaction",
    "phim ngắn", "short film", "phim hài", "phim tình cảm",
    "tiktok", "shorts", "funny", "hài",
    "lồng tiếng", "thuyết minh", "vietsub", "trung quốc",
    "truyện", "tập", "drama", "series", "anime",
]

NEWS_CHANNELS = [
    "VTV24", "VTV1", "VTV", "VTC", "THVL", "HTV",
    "Truyền Hình Nhân Dân", "Báo Thanh Niên", "Tuổi Trẻ",
    "VnExpress", "Dân Trí", "Đài Truyền Hình",
]

def is_relevant_video(title, channel=""):
    title_lower = title.lower()
    channel_lower = channel.lower()

    for bad in TITLE_MUST_NOT_CONTAIN:
        if bad.lower() in title_lower:
            return False

    is_news_channel = any(ch.lower() in channel_lower for ch in NEWS_CHANNELS)
    has_news_keyword = any(kw.lower() in title_lower for kw in TITLE_MUST_CONTAIN)

    return is_news_channel and has_news_keyword

# =====================================================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

def _random_ua():
    return random.choice(USER_AGENTS)

def _human_delay(min_s=3, max_s=8):
    time.sleep(random.uniform(min_s, max_s))

def _make_common_opts():
    return {
        'cookiefile': 'cookies.txt',
        'extractor_args': {'youtube': {'player_client': ['web']}},
        'http_headers': {
            'User-Agent': _random_ua(),
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        },
    }


def iter_video_ids_multi_query(queries=None, max_pages_per_query=5):
    if queries is None:
        queries = QUERIES

    shuffled = queries.copy()
    random.shuffle(shuffled)
    seen = set()

    for query in shuffled:
        print(f"\n{'='*50}")
        print(f"Query: '{query}'")
        print(f"{'='*50}")

        page_token = None

        for page_num in range(1, max_pages_per_query + 1):
            try:
                params = dict(
                    q=query, part='id,snippet', type='video',
                    videoDuration='long', maxResults=50,
                    relevanceLanguage='vi', regionCode='VN',
                )
                if page_token:
                    params['pageToken'] = page_token

                response = youtube.search().list(**params).execute()
                items = response.get('items', [])

                if not items:
                    break

                print(f"  [Trang {page_num}] {len(items)} video")

                for item in items:
                    vid_id  = item['id']['videoId']
                    title   = item['snippet']['title']
                    channel = item['snippet'].get('channelTitle', '')

                    if vid_id in seen:
                        continue
                    seen.add(vid_id)

                    if not is_relevant_video(title, channel):
                        print(f"   x Bo: {title[:50]} [{channel}]")
                        continue

                    print(f"   v {title[:50]} [{channel}] ({vid_id})")
                    yield vid_id

                page_token = response.get('nextPageToken')
                if not page_token:
                    break

                time.sleep(random.uniform(1.0, 2.5))

            except Exception as e:
                print(f"  [Loi API] {e}")
                break

        wait = random.uniform(5, 12)
        print(f"\n  Nghi {wait:.1f}s truoc query tiep theo...")
        time.sleep(wait)


def download_worst_video_for_test(vid_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{vid_id}_test.mp4')
    if os.path.exists(out_path):
        return out_path

    opts = _make_common_opts()
    ydl_opts = {
        **opts,
        'format': 'worstvideo[ext=mp4]/worst[ext=mp4]/worstvideo/worst/best',
        'outtmpl': out_path,
        'quiet': True, 'no_warnings': True,
    }
    if FFMPEG_PATH:
        ydl_opts['ffmpeg_location'] = FFMPEG_PATH

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f'https://www.youtube.com/watch?v={vid_id}', download=False)
            formats = info.get('formats', [])
            if not any(f.get('vcodec', 'none') != 'none' for f in formats):
                print(f"  [Bo qua] {vid_id} khong co format video")
                return None
            ydl.download([f'https://www.youtube.com/watch?v={vid_id}'])
        _human_delay(2, 5)
        return out_path
    except Exception as e:
        print(f"  [Loi test download] {vid_id}: {e}")
        return None


def download_video_full(vid_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{vid_id}.mp4')
    if os.path.exists(out_path):
        return out_path

    opts = _make_common_opts()
    ydl_opts = {
        **opts,
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_path,
        'quiet': True,
    }
    if FFMPEG_PATH:
        ydl_opts['ffmpeg_location'] = FFMPEG_PATH

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={vid_id}'])
        _human_delay(4, 10)
        return out_path
    except Exception as e:
        print(f"  [Loi full download] {vid_id}: {e}")
        return None