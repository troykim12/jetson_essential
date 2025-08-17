import os, time
import numpy as np
import jetson_utils_python as jetson_utils
import jetson_inference
import vlc
import subprocess
from uploader import upload_file_to, unique_name, CONT_FULL, CONT_CROP, SAS_FULL, SAS_CROP
from datetime import datetime, timezone
import certifi
from pymongo import MongoClient


HOST = os.getenv("COSMOS_HOST")
USER = os.getenv("COSMOS_USER")
PASS = os.getenv("COSMOS_PASS")
DB_NAME = os.getenv("COSMOS_DB")
COL_NAME = os.getenv("COSMOS_COL")

URI = f"mongodb://{USER}:{PASS}@{HOST}:10255/?ssl=true&replicaSet=globaldb&retrywrites=false"

client = MongoClient(
    URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=8000
)
db = client[DB_NAME]
col = db[COL_NAME]

def make_blob_name(label, crop=True):
    """
    label: 객체 라벨 (예: 'person', 'dog')
    crop: True면 crop_xxx, False면 full_xxx
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "crop" if crop else "full"
    return f"{prefix}_{label}_{ts}.jpg"

def sanitize_label(s: str) -> str:
    # 공백/대문자/특수문자 정리 → 파일명 안전
    s = s.strip().lower().replace(" ", "_")
    return "".join(ch for ch in s if (ch.isalnum() or ch in ("_", "-")))

def label_prefix(label: str, crop: bool) -> str:
    # 기존 함수들이 뒤에서 "_<UTC타임스탬프>.jpg"를 붙이므로 여기서는 prefix만 돌려줌
    return f"{'crop' if crop else 'full'}_{label}"

# =========================
# Box 변환 유틸
# =========================
def tlbr_to_xyah(tlbr):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]
    cx = tlbr[0] + w * 0.5
    cy = tlbr[1] + h * 0.5
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyah_to_tlbr(xyah):
    cx, cy, a, h = xyah
    w = a * h
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# ── 박스 스타일: 'outline' = 테두리, 'yolo' = 코너 강조
STYLE = "yolo"   # 필요시 'outline' 로 변경

def draw_rect_outline(img, x1,y1,x2,y2, color=(0,255,0,255)):
    jetson_utils.cudaDrawRect(img, (int(x1), int(y1), int(x2), int(y2)), color)

def draw_yolo_corners(img, x1,y1,x2,y2, color=(0,255,0,255), k=0.25, t=2):
    x1, y1, x2, y2 = map(int, (x1,y1,x2,y2))
    w, h = x2 - x1, y2 - y1
    L = int(min(w, h) * k)
    for i in range(t):
        # TL
        jetson_utils.cudaDrawLine(img, (x1, y1+i),   (x1+L, y1+i),   color)
        jetson_utils.cudaDrawLine(img, (x1+i, y1),   (x1+i, y1+L),   color)
        # TR
        jetson_utils.cudaDrawLine(img, (x2-L, y1+i), (x2,   y1+i),   color)
        jetson_utils.cudaDrawLine(img, (x2-1-i, y1), (x2-1-i, y1+L), color)
        # BL
        jetson_utils.cudaDrawLine(img, (x1, y2-1-i), (x1+L, y2-1-i), color)
        jetson_utils.cudaDrawLine(img, (x1+i, y2-L), (x1+i, y2),     color)
        # BR
        jetson_utils.cudaDrawLine(img, (x2-L, y2-1-i), (x2,   y2-1-i), color)
        jetson_utils.cudaDrawLine(img, (x2-1-i, y2-L), (x2-1-i, y2),   color)

def draw_box(img, x1,y1,x2,y2, color=(0,255,0,255)):
    if STYLE == "outline":
        draw_rect_outline(img, x1,y1,x2,y2, color)
    else:
        draw_yolo_corners(img, x1,y1,x2,y2, color)

def id2color(tid):
    np.random.seed(tid)  # 트랙별 고정 색상
    r,g,b = np.random.randint(80, 255, 3).tolist()
    return (int(r), int(g), int(b), 255)


# =========================
# IoU & Mahalanobis
# =========================
def iou_xyxy(a, b):
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)

def mahalanobis_squared(y, S_inv):
    return np.einsum("ki,ij,kj->k", y, S_inv, y)

# =========================
# 칼만 필터 (SORT/ByteTrack 스타일)
# =========================
class KalmanFilterXYAH:
    def __init__(self, dt=1.0):
        self.dt = dt
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i+4] = dt
        self._H = np.zeros((4, 8), dtype=np.float32)
        self._H[0,0] = self._H[1,1] = self._H[2,2] = self._H[3,3] = 1.0
        q_pos, q_vel = 1.0, 10.0
        self._Q = np.diag([q_pos, q_pos, 1e-2, q_pos, q_vel, q_vel, 1e-3, q_vel]).astype(np.float32)
        self._R = np.diag([1.0, 1.0, 1e-2, 1.0]).astype(np.float32)
        self._I8 = np.eye(8, dtype=np.float32)

    def initiate(self, xyah):
        mean = np.zeros((8,), dtype=np.float32)
        mean[:4] = xyah
        P = np.diag([10, 10, 1e-1, 10, 100, 100, 1e-2, 100]).astype(np.float32)
        return mean, P

    def predict(self, mean, P):
        # mean = self._F @ mean
        mean = np.dot(self._F, mean)
        # P = self._F @ P @ self._F.T + self._Q
        P = np.dot(np.dot(self._F, P), self._F.T) + self._Q
        return mean, P

    def project(self, mean, P):
        # S = self._H @ P @ self._H.T + self._R
        S = np.dot(np.dot(self._H, P), self._H.T) + self._R
        # z = self._H @ mean
        z = np.dot(self._H, mean)
        return z, S

    def update(self, mean, P, z_obs):
        z_pred, S = self.project(mean, P)
        # K = P @ self._H.T @ np.linalg.inv(S)
        # 더 안정적으로: solve(S, (H P)^T).T
        HP = np.dot(self._H, P)           # (4x8)
        K  = np.linalg.solve(S, HP).T     # (8x4)
        # mean = mean + K @ (z_obs - z_pred)
        mean = mean + np.dot(K, (z_obs - z_pred))
        # P = (self._I8 - K @ self._H) @ P
        KH = np.dot(K, self._H)
        P  = np.dot((self._I8 - KH), P)
        return mean, P
# =========================
# 트랙 / 트래커 (ByteTrack-KF 경량)
# =========================
class TrackKF:
    __slots__ = ("mean","cov","score","id","age","miss","hit","class_id","activated","kf")
    def __init__(self, xyah, score, class_id, tid, kf: KalmanFilterXYAH):
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyah)
        self.score = float(score)
        self.class_id = int(class_id)
        self.id = tid
        self.age = 0
        self.miss = 0
        self.hit = 1
        self.activated = True

    @property
    def tlbr(self):
        return xyah_to_tlbr(self.mean[:4])

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.miss += 1

    def update(self, tlbr, score):
        xyah = tlbr_to_xyah(tlbr)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyah)
        self.score = float(score)
        self.hit += 1
        self.miss = 0
        self.activated = True

class BYTETrackerKF:
    def __init__(self, track_thresh=0.5, match_thresh=0.7, match_thresh_low=0.5,
                 buffer_ttl=30, min_box_area=10, gate_maha=True, maha_th=25.0,
                 dt=1.0, max_ids=1<<30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_low = match_thresh_low
        self.buffer_ttl = buffer_ttl
        self.min_box_area = min_box_area
        self.gate_maha = gate_maha
        self.maha_th = maha_th
        self.kf = KalmanFilterXYAH(dt=dt)
        self.tracks = []
        self._next_id = 1
        self._max_ids = max_ids

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        if self._next_id >= self._max_ids:
            self._next_id = 1
        return tid

    @staticmethod
    def _greedy_match(iou_mat, iou_th):
        matches, u_a, u_b = [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
        if iou_mat.size == 0:
            return matches, u_a, u_b
        iou_copy = iou_mat.copy()
        while True:
            maxv = iou_copy.max() if iou_copy.size else 0.0
            if maxv < iou_th or maxv <= 0: break
            i, j = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
            matches.append((int(i), int(j)))
            iou_copy[i, :], iou_copy[:, j] = -1.0, -1.0
        matched_a = {m[0] for m in matches}
        matched_b = {m[1] for m in matches}
        u_a = [i for i in range(iou_mat.shape[0]) if i not in matched_a]
        u_b = [j for j in range(iou_mat.shape[1]) if j not in matched_b]
        return matches, u_a, u_b

    def _gating_mask(self, dets_xyah, tracks):
        if not self.gate_maha or len(tracks) == 0 or len(dets_xyah) == 0:
            return np.ones((len(tracks), len(dets_xyah)), dtype=bool)
        mask = np.zeros((len(tracks), len(dets_xyah)), dtype=bool)
        for i, t in enumerate(tracks):
            z_pred, S = self.kf.project(t.mean, t.cov)
            S_inv = np.linalg.inv(S)
            y = dets_xyah - z_pred[None, :]
            d2 = mahalanobis_squared(y, S_inv)
            mask[i] = d2 < self.maha_th
        return mask

    def update(self, dets_tlbr_scores, class_ids=None):
        for t in self.tracks:
            t.predict()

        if dets_tlbr_scores is None or len(dets_tlbr_scores) == 0:
            self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
            return [t for t in self.tracks if t.miss == 0 and t.activated]

        dets = np.asarray(dets_tlbr_scores, dtype=np.float32)
        cls_arr = np.zeros((dets.shape[0],), dtype=np.int32) if class_ids is None else np.asarray(class_ids, dtype=np.int32)

        high_mask = dets[:, 4] >= self.track_thresh
        dets_high, dets_low = dets[high_mask], dets[~high_mask]
        cls_high, cls_low  = cls_arr[high_mask], cls_arr[~high_mask]

        active_idx = list(range(len(self.tracks)))
        tr_tlbr = np.array([self.tracks[i].tlbr for i in active_idx], dtype=np.float32) if active_idx else np.zeros((0,4), np.float32)
        iou_mat = iou_xyxy(tr_tlbr, dets_high[:, :4]) if dets_high.size else np.zeros((tr_tlbr.shape[0], 0), np.float32)

        if dets_high.size and len(self.tracks):
            dets_high_xyah = np.stack([tlbr_to_xyah(b) for b in dets_high[:, :4]], axis=0)
            gate = self._gating_mask(dets_high_xyah, [self.tracks[i] for i in active_idx])
            iou_mat = np.where(gate, iou_mat, -1.0)
        matches, u_tr, u_dt = self._greedy_match(iou_mat, self.match_thresh)

        for (ti, di) in matches:
            t = self.tracks[active_idx[ti]]
            t.update(dets_high[di, :4], dets_high[di, 4])
            t.class_id = int(cls_high[di])

        if len(u_tr) and len(dets_low):
            tr_rest = np.array([self.tracks[active_idx[i]].tlbr for i in u_tr], dtype=np.float32)
            iou_low = iou_xyxy(tr_rest, dets_low[:, :4])
            if len(self.tracks):
                dets_low_xyah = np.stack([tlbr_to_xyah(b) for b in dets_low[:, :4]], axis=0)
                gate2 = self._gating_mask(dets_low_xyah, [self.tracks[active_idx[i]] for i in u_tr])
                iou_low = np.where(gate2, iou_low, -1.0)
            matches2, u_tr2, u_dt2 = self._greedy_match(iou_low, self.match_thresh_low)
            for (ti2, di2) in matches2:
                t = self.tracks[active_idx[u_tr[ti2]]]
                t.update(dets_low[di2, :4], dets_low[di2, 4])
                t.class_id = int(cls_low[di2])
            u_tr_final = [u_tr[i] for i in u_tr2]
            u_dt_high_final = [i for i in range(len(dets_high)) if i not in u_dt]
            u_dt_low_final  = u_dt2
        else:
            u_tr_final = u_tr
            u_dt_high_final = u_dt
            u_dt_low_final  = list(range(len(dets_low)))

        for di in u_dt_high_final:
            tlbr = dets_high[di, :4]
            if (tlbr[2]-tlbr[0])*(tlbr[3]-tlbr[1]) >= self.min_box_area:
                xyah = tlbr_to_xyah(tlbr)
                tid = self._new_id()
                self.tracks.append(TrackKF(xyah, dets_high[di, 4], int(cls_high[di]), tid, self.kf))

        self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
        return [t for t in self.tracks if t.miss == 0 and t.activated]

# ============== 오디오 설정 ==============
MP3_FILE = "/home/jetson/vigil/ji/lion_roar_3sec.mp3"
AUDIO_LEN_SEC = 3.2
assert os.path.exists(MP3_FILE), f"MP3 not found: {MP3_FILE}"
vlc_instance = vlc.Instance('--no-xlib', '--no-video', '--aout=pulse')
player = vlc_instance.media_player_new()
def play_mp3_once():
    media = vlc_instance.media_new(MP3_FILE)
    player.set_media(media)
    player.stop()
    player.play()
    print("------------------------playing---------------------")
    time.sleep(0.05)

# ============== 라벨/환경 ==============
WIDTH, HEIGHT = 640, 480
LABELS_PATH = "/home/jetson/vigil/ji/labels.txt"
TARGET_NAME = "dog"

def normalize(s):
    return " ".join(s.strip().lower().replace("_", " ").split())


label_to_id = {}
labels_list = []
with open(LABELS_PATH, "r") as f:
    for idx, line in enumerate(f):
        name = line.strip()
        if name:
            labels_list.append(name)
            label_to_id[normalize(name)] = idx

id_to_label = {i: name for i, name in enumerate(labels_list)}

target_id = None
for cand in [TARGET_NAME, "dog"]:
    key = normalize(cand)
    if key in label_to_id:
        target_id = label_to_id[key]; break
if target_id is None:
    print(f"[WARN] '{TARGET_NAME}' not in labels. first keys: {list(label_to_id.keys())[:10]}")

# ============== 카메라/표시/모델 ==============
camera = jetson_utils.gstCamera(WIDTH, HEIGHT, "/dev/video0")
display_local = jetson_utils.videoOutput()
# display_rtmp  = jetson_utils.videoOutput("rtmp://192.168.0.3/live/jetson1")
display_rtmp  = jetson_utils.videoOutput("rtmp://20.249.68.101/live/jetson1")
# display_rtmp  = jetson_utils.videoOutput("rtmp://agrilook-be-stream.koreacentral.cloudapp.azure.com:1935/live/jetson1")

net = jetson_inference.detectNet(argv=[
    "--model=/home/jetson/vigil/ji/mb2-ssd-lite_320.onnx",
    f"--labels={LABELS_PATH}",
    "--input-blob=input_0",
    "--output-cvg=scores",
    "--output-bbox=boxes",
    "--threshold=0.3",
    "--verbose"
])

# ============== 트래커/폰트/캡쳐 설정 ==============
tracker = BYTETrackerKF(track_thresh=0.5, match_thresh=0.7, match_thresh_low=0.5,
                        buffer_ttl=30, min_box_area=12, gate_maha=True, maha_th=25.0,
                        dt=1.0)
font = jetson_utils.cudaFont()

# ---- 캡쳐 관련 파라미터 ----
CONF_TH = 0.6      #
PERSIST_SEC = 3.0      # 
CAPTURE_DIR = "/home/jetson/vigil/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# 트랙 가시 시간/캡쳐 관리
seen_since = {}        # {track_id: first_seen_time}
captured_ids = set()   # 이미 캡쳐된 track_id(중복 캡쳐 방지)

def save_frame(img, path):
    # CUDA 메모리 직접 저장 (jetson-utils), 실패 시 OpenCV fallback
    try:
        jetson_utils.saveImage(path, img)
    except Exception:
        try:
            import cv2
            arr = jetson_utils.cudaToNumpy(img, HEIGHT, WIDTH, 4)  # RGBA
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(path, bgr)
        except Exception as e:
            print(f"[WARN] save_frame fallback failed: {e}")

def crop_save_and_upload(img, bbox_xyxy, prefix="intrusion", persist_dir=CAPTURE_DIR):
    """
    bbox_xyxy=(x1,y1,x2,y2) 기준으로 GPU에서 바로 크롭해서
    1) persist_dir에 JPEG로 영구 저장
    2) 같은 파일명을 사용해 Azure Blob에도 업로드
    반환: (local_path, blob_url)
    """
    import datetime as dt
    os.makedirs(persist_dir, exist_ok=True)

    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = int(x2); y2 = int(y2)
    w = max(1, x2 - x1); h = max(1, y2 - y1)

    # 1) GPU 크롭
    try:
        fmt = getattr(img, 'format', 'rgba32f')
        crop_gpu = jetson_utils.cudaAllocMapped(width=w, height=h, format=fmt)
        jetson_utils.cudaCrop(img, crop_gpu, (x1, y1, x2, y2))
    except Exception as e:
        print(f"[WARN] cudaCrop failed, fallback to full frame. err={e}")
        crop_gpu = img  # 실패 시 전체 프레임 저장 (최후의 수단)

    # 2) 로컬(영구) 저장 경로 생성
    ts = dt.datetime.utcnow().isoformat().replace(":", "-")
    filename = f"{prefix}_{ts}.jpg"
    local_path = os.path.join(persist_dir, filename)

    # 3) JPEG 저장 (확장자에 따라 자동 인코딩)
    jetson_utils.saveImage(local_path, crop_gpu)

    # 4) Blob 업로드 (같은 파일명으로)
    try:
        blob_url = upload_file_to(local_path, filename,
                                  container=CONT_CROP, sas_url=SAS_CROP)
        print(f"[BLOB] uploaded: {blob_url}")
    except Exception as e:
        print(f"[BLOB] upload failed: {e}")
        blob_url = ""

    return local_path, blob_url

def full_save_and_upload(img, prefix="full", persist_dir=CAPTURE_DIR):
    import datetime as dt
    os.makedirs(persist_dir, exist_ok=True)

    ts = dt.datetime.utcnow().isoformat().replace(":", "-")
    filename = f"{prefix}_{ts}.jpg"
    local_path = os.path.join(persist_dir, filename)

    # 입력 포맷 그대로 저장 (필요하면 rgba8 변환 추가 가능)
    jetson_utils.saveImage(local_path, img)

    # === 전체 프레임 컨테이너로 업로드 ===
    try:
        blob_url = upload_file_to(local_path, filename,
                                  container=CONT_FULL, sas_url=SAS_FULL)
        print(f"[BLOB-FULL] uploaded: {blob_url}")
    except Exception as e:
        print(f"[BLOB-FULL] upload failed: {e}")
        blob_url = ""

    return local_path, blob_url

print("=== Starting ===")
print("DISPLAY:", os.environ.get("DISPLAY"))
print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))

# ============== 타이머/오디오 상태 ==============
present_since = None
audio_playing = False
audio_started = 0.0
start = time.time()

try:
    while True:
        img, w, h = camera.CaptureRGBA()
        if img is None:
            print("Capture failed"); break

        # 1) Detect
        dets = net.Detect(img, w, h, overlay='none')

        # 2) target 클래스 + 최소 점수 필터 → 트래커 입력
        det_list, cls_list = [], []
        for d in dets:
            if d.Confidence < CONF_TH:
                continue
            x1, y1, x2, y2 = float(d.Left), float(d.Top), float(d.Right), float(d.Bottom)
            score = float(d.Confidence)
            det_list.append([x1, y1, x2, y2, score])
            cls_list.append(int(d.ClassID))
        det_arr = np.array(det_list, dtype=np.float32) if len(det_list) else np.zeros((0,5), np.float32)

        tracks = tracker.update(det_arr, class_ids=np.array(cls_list) if len(cls_list) else None)

        # 3) 오버레이 + 3초 지속 체크
        now = time.time()
        active_ids = set()
        for t in tracks:
            tid = t.id
            active_ids.add(tid)

            # 오버레이
            x1, y1, x2, y2 = t.tlbr
            color = id2color(tid)          # 트랙 ID별 색상
            draw_box(img, x1, y1, x2, y2, color)

            # first_seen 기록/업데이트
            if tid not in seen_since:
                seen_since[tid] = now
            elapsed = now - seen_since[tid]

            # 캡션에 경과시간 표시
            font.OverlayText(img, w, h, f"id:{tid} s:{t.score:.2f} t:{elapsed:.1f}s",
                             int(x1), max(0, int(y1)-18), font.White, font.Gray40)

            # 3초 이상 연속 유지 & 아직 미캡쳐 → 프레임 저장
            if elapsed >= PERSIST_SEC and tid not in captured_ids:
                # 트랙의 class_id → labels.txt에서 라벨 얻기
                label_raw = id_to_label.get(int(t.class_id), str(t.class_id))
                label = sanitize_label(label_raw)

                # 1) 크롭 → intrusion-crop-images (crop_<label>_YYYYmmdd_HHMMSS.jpg)
                crop_local, crop_url = crop_save_and_upload(
                    img, (x1, y1, x2, y2),
                    prefix=label_prefix(label, crop=True)
                )

                # 2) 원본 전체 → intrusion-images (full_<label>_YYYYmmdd_HHMMSS.jpg)
                full_local, full_url = full_save_and_upload(
                    img,
                    prefix=label_prefix(label, crop=False)
                )

                print(f"[CAPTURE] crop: {crop_local} | {crop_url}")
                print(f"[CAPTURE] full: {full_local} | {full_url}")

		        # === DB로 클래스와 시간(UTC)만 전송 ===
                try:
                    col.insert_one({
                        "class": label,
                        "datetime": datetime.now(timezone.utc)  # BSON Date로 저장
                    })
                    print(f"[DB] inserted class={label}")
                except Exception as e:
                    print(f"[DB] insert failed: {e}")

                captured_ids.add(tid)


        # 프레임에서 사라진 ID는 가시시간 리셋(다시 3초 조건을 만족해야 캡쳐)
        for tid in list(seen_since.keys()):
            if tid not in active_ids:
                del seen_since[tid]
        # (captured_ids는 유지: 같은 트랙 ID로 이미 저장했으면 중복 방지. 새 트랙이면 ID가 달라짐)

        # 4) 2초 연속 감지(트랙 존재 기반) → 사운드
        is_present = len(tracks) > 0
        if is_present:
            if present_since is None:
                present_since = now
            elif (now - present_since) >= 3.0 and not audio_playing:
                play_mp3_once()
                audio_started = time.time()
                audio_playing = True
        else:
            present_since = None

        # 5) 오디오 종료
        if audio_playing and (time.time() - audio_started) >= AUDIO_LEN_SEC:
            player.stop()
            audio_playing = False
            
        # 6) 출력
        if display_local and display_local.IsStreaming():
            display_local.Render(img)
            display_local.SetStatus(f"{net.GetNetworkFPS():.0f} FPS | det={len(dets)} | tracks={len(tracks)}")
        if display_rtmp and display_rtmp.IsStreaming():
            display_rtmp.Render(img)

        if time.time() - start > 100:
            print("====================Timeout exit============================="); break

except KeyboardInterrupt:
    print("Interrupted")
finally:
    try: camera.Close()
    except: pass
    try:
        if display_local: display_local.Close()
    except: pass
    try:
        if display_rtmp: display_rtmp.Close()
    except: pass
    try: player.stop()
    except: pass
    print("Done.")
