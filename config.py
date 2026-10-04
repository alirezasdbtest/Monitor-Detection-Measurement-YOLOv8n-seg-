"""تنظیمات فاین‌تیون مانیتور روی تصاویر CCTV / RTSP."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent

CCTV_DIR = PROJECT / "cctv"
CCTV_VIDEOS = CCTV_DIR / "raw_videos"
CCTV_FRAMES = CCTV_DIR / "raw_frames"
CCTV_ANNOTATED = CCTV_DIR / "annotated"      # images/ + labels/ (فقط بازبینی‌شده‌ها)
CCTV_UNREVIEWED = CCTV_DIR / "unreviewed"    # فریم‌هایی که prefill چیزی ندید؛ باید دستی بررسی شوند
CCTV_NEGATIVES = CCTV_DIR / "negatives"      # فقط فریم‌هایی که هیچ مانیتوری ندارند
CCTV_DS = PROJECT / "cctv_ds"

RUNS = PROJECT / "runs"
MODELS = PROJECT / "models"
OUTPUTS = PROJECT / "outputs"

# همان RTSP که در camera monitor.py استفاده می‌شود (در urls.txt هم می‌توانید بگذارید)
#DEFAULT_RTSP = "rtsp://user:Aa123456@192.168.1.232:554"
#DEFAULT_RTSP = "rtsp://admin:ms123456@192.168.1.190:554"
DEFAULT_RTSP = "rtsp://user:Aa123456@192.168.1.233:554"

MONITOR_CLASS_NAMES = ["monitor"]

# id کلاس مانیتور در مدل «اصلی» (برای prefill). None = تشخیص خودکار از روی نام کلاس.
MONITOR_SOURCE_CLASS_ID = None
MONITOR_NAME_ALIASES = {"monitor", "tv", "screen", "display"}

WEIGHT_CANDIDATES = [
    MODELS / "best.pt",
    ROOT / ".myvenv" / "best.pt",
    Path(r"C:\Users\h.akbari\Desktop\Saidi_proj\proj\data\runs\segment\runs\segment\industryshapes_v1-2\weights\best.pt"),
    Path(r"C:\Users\h.akbari\Desktop\Saidi_proj\proj\runs\quick\weights\best.pt"),
    Path("yolov8n-seg.pt"),
]

# --- استخراج فریم ---
EXTRACT_EVERY = 150        # هر N فریم یک نمونه
EXTRACT_MAX = 400          # حداکثر فریم از هر منبع
EXTRACT_DIFF_THR = 6.0     # آستانه‌ی تفاوت میانگین (۰-۲۵۵)؛ اگر فریم کم شد 4 بگذار

# --- دیتاست ---
VAL_RATIO = 0.2
VAL_GAP = 4                # تعداد فریم‌های حائل بین train و val (حذف می‌شوند)
REPLAY_DIR = None          # مثلاً Path(r"...\\original_ds") با images/ و labels/
REPLAY_RATIO = 0.25        # سهم داده اصلی از train نهایی

# --- فاین‌تیون ---
FINETUNE_EPOCHS = 40
FINETUNE_IMGSZ = 640       # اگر مانیتورها کوچک‌اند 960 تست کن
FINETUNE_LR0 = 1e-4
FINETUNE_OPTIMIZER = "AdamW"
FINETUNE_FREEZE = -1       # -1 = خودکار (اگر train < 300 تصویر: 10 لایه) | 0 = بدون freeze | N
SEED = 0

PREDICT_CONF = 0.45
PREDICT_IOU = 0.45