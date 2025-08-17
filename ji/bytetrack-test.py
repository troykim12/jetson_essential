import os, time
import numpy as np
import jetson_utils_python as jetson_utils
import jetson_inference
import vlc
# =========================
# ByteTrack-lite (pure NumPy)
# =========================
def iou_xyxy(a, b):
    # a: (N,4) [x1,y1,x2,y2], b: (M,4)
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])     # (N,M,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])     # (N,M,2)
    wh = np.clip(rb - lt, 0, None)                      # (N,M,2)
    inter = wh[..., 0] * wh[..., 1]                     # (N,M)
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])      # (N,)
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])      # (M,)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)
class Track:
    __slots__ = ("tlbr","score","id","age","miss","hit","class_id","activated")
    def __init__(self, tlbr, score, class_id, tid):
        self.tlbr = tlbr.astype(np.float32)   # [x1,y1,x2,y2]
        self.score = float(score)
        self.class_id = int(class_id)
        self.id = tid
        self.age = 0
        self.miss = 0
        self.hit = 1
        self.activated = True
    def update(self, tlbr, score):
        self.tlbr = tlbr.astype(np.float32)
        self.score = float(score)
        self.hit += 1
        self.miss = 0
        self.age += 1
        self.activated = True
    def predict(self):
        # Kalman 없이 정지 가정(라이트 버전). 필요하면 이동 평균 등 추가 가능.
        self.age += 1
        self.miss += 1
class BYTETrackerLite:
    def __init__(self, track_thresh=0.5, match_thresh=0.7, match_thresh_low=0.5,
                 buffer_ttl=30, min_box_area=10, max_ids=1<<30):
        self.track_thresh = track_thresh           # 트랙 생성/유지에 사용할 상위 점수 기준
        self.match_thresh = match_thresh           # 1단계 매칭 IoU 임계값
        self.match_thresh_low = match_thresh_low   # 2단계(저점수) 매칭 IoU 임계값
        self.buffer_ttl = buffer_ttl               # miss 허용 프레임 수
        self.min_box_area = min_box_area
        self.tracks = []       # 활성+버퍼 트랙
        self._next_id = 1
        self._max_ids = max_ids
    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        if self._next_id >= self._max_ids:  # 안전 가드
            self._next_id = 1
        return tid
    @staticmethod
    def _greedy_match(iou_mat, iou_th):
        # 그리디: 가장 IoU 큰 쌍부터 배정
        matches, u_a, u_b = [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
        if iou_mat.size == 0:
            return matches, u_a, u_b
        iou_copy = iou_mat.copy()
        while True:
            i = j = -1
            maxv = iou_copy.max() if iou_copy.size else 0.0
            if maxv < iou_th or maxv <= 0:
                break
            idx = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
            i, j = int(idx[0]), int(idx[1])
            matches.append((i, j))
            # 배정된 행/열 제거(음수로 설정)
            iou_copy[i, :] = -1.0
            iou_copy[:, j] = -1.0
        matched_a = {m[0] for m in matches}
        matched_b = {m[1] for m in matches}
        u_a = [i for i in range(iou_mat.shape[0]) if i not in matched_a]
        u_b = [j for j in range(iou_mat.shape[1]) if j not in matched_b]
        return matches, u_a, u_b
    def update(self, dets_tlbr_scores, class_ids=None):
        """
        dets_tlbr_scores: (K,5) [x1,y1,x2,y2,score]
        class_ids: (K,) or None
        return: list of active tracks (with miss==0)
        """
        if dets_tlbr_scores is None or len(dets_tlbr_scores) == 0:
            # 예측/삭제만
            for t in self.tracks:
                t.predict()
            self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
            return [t for t in self.tracks if t.miss == 0 and t.activated]
        dets = np.asarray(dets_tlbr_scores, dtype=np.float32)
        if class_ids is None:
            cls_arr = np.zeros((dets.shape[0],), dtype=np.int32)
        else:
            cls_arr = np.asarray(class_ids, dtype=np.int32)
        # split high / low score
        high_mask = dets[:, 4] >= self.track_thresh
        dets_high = dets[high_mask]
        dets_low  = dets[~high_mask]
        cls_high  = cls_arr[high_mask]
        cls_low   = cls_arr[~high_mask]
        # 1) 예측(라이트)
        for t in self.tracks:
            t.predict()
        # 2) 1단계: high-score 매칭
        active_idx = [i for i, t in enumerate(self.tracks)]
        tr_tlbr = np.array([self.tracks[i].tlbr for i in active_idx], dtype=np.float32) if active_idx else np.zeros((0,4), np.float32)
        iou_mat = iou_xyxy(tr_tlbr, dets_high[:, :4]) if dets_high.size else np.zeros((tr_tlbr.shape[0], 0), np.float32)
        matches, u_tr, u_dt = self._greedy_match(iou_mat, self.match_thresh)
        # 업데이트
        for (ti, di) in matches:
            t = self.tracks[active_idx[ti]]
            t.update(dets_high[di, :4], dets_high[di, 4])
            t.class_id = int(cls_high[di])
        # 3) 2단계: 남은 트랙 vs low-score 디텍션 매칭(좀 더 낮은 IoU 허들)
        if len(u_tr) and len(dets_low):
            tr_rest = np.array([self.tracks[active_idx[i]].tlbr for i in u_tr], dtype=np.float32)
            iou_low = iou_xyxy(tr_rest, dets_low[:, :4])
            matches2, u_tr2, u_dt2 = self._greedy_match(iou_low, self.match_thresh_low)
            # 매칭된 것만 갱신
            for k, (ti2, di2) in enumerate(matches2):
                t = self.tracks[active_idx[u_tr[ti2]]]
                t.update(dets_low[di2, :4], dets_low[di2, 4])
                t.class_id = int(cls_low[di2])
            # 2단계에서도 못 맞춘 트랙은 u_tr2, 디텍은 u_dt2
            u_tr_final = [u_tr[i] for i in u_tr2]
            u_dt_high_final = [i for i in range(len(dets_high)) if i not in u_dt]  # 이미 처리된 것 제외 정보용
            u_dt_low_final  = u_dt2
        else:
            u_tr_final = u_tr
            u_dt_low_final = list(range(len(dets_low)))  # 전부 미매칭
            u_dt_high_final = u_dt
        # 4) 새 트랙 생성: high-score에서 미매칭된 디텍션
        new_tracks = []
        for di in u_dt_high_final:
            tlbr = dets_high[di, :4]
            if (tlbr[2]-tlbr[0])*(tlbr[3]-tlbr[1]) >= self.min_box_area:
                new_tracks.append(Track(tlbr, dets_high[di, 4], int(cls_high[di]), self._new_id()))
        self.tracks.extend(new_tracks)
        # 5) 오래 실종된 트랙 제거
        self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
        # 활성 트랙(금방 업데이트된 프레임 기준)
        active_tracks = [t for t in self.tracks if t.miss == 0 and t.activated]
        return active_tracks
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
    print("------------------playing-----------------")
    time.sleep(0.05)
# ============== 라벨 매핑/환경 ==============
WIDTH, HEIGHT = 640, 480
LABELS_PATH = "/home/jetson/vigil/ji/labels.txt"
TARGET_NAME = "dog"
def normalize(s):
    return " ".join(s.strip().lower().replace("_", " ").split())
label_to_id = {}
with open(LABELS_PATH, "r") as f:
    for idx, line in enumerate(f):
        name = line.strip()
        if name:
            label_to_id[normalize(name)] = idx
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
display_rtmp  = jetson_utils.videoOutput("rtmp://192.168.68.87/live/jetson1")
net = jetson_inference.detectNet(argv=[
    "--model=/home/jetson/vigil/ji/mb2-ssd-lite_300.onnx",
    f"--labels={LABELS_PATH}",
    "--input-blob=input_0",
    "--output-cvg=scores",
    "--output-bbox=boxes",
    "--threshold=0.3",
    "--verbose"
])
# ============== 트래커/오버레이 ==============
# ByteTrack 기본값: score 0.5 이상은 1차 매칭/생성, IoU 0.7/0.5 허들
tracker = BYTETrackerLite(track_thresh=0.5, match_thresh=0.7, match_thresh_low=0.5,
                          buffer_ttl=30, min_box_area=12)
font = jetson_utils.cudaFont()  # 트랙 ID 텍스트용
print("=== Starting ===")
print("DISPLAY:", os.environ.get("DISPLAY"))
print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))
# ============== 타이머/오디오 상태 ==============
present_since = None        # 타겟(트랙 단위)이 보이기 시작한 시각
audio_playing = False
audio_started = 0.0
start = time.time()
try:
    while True:
        img, w, h = camera.CaptureRGBA()
        if img is None:
            print("Capture failed"); break
        # 1) 디텍션
        dets = net.Detect(img, w, h)
        # 2) 대상 클래스만 모아 ByteTrack 업데이트
        det_list = []
        cls_list = []
        for d in dets:
            if target_id is not None and d.ClassID != target_id:
                continue
            # jetson-inference bbox: Left, Top, Right, Bottom (float)
            x1, y1, x2, y2 = float(d.Left), float(d.Top), float(d.Right), float(d.Bottom)
            score = float(d.Confidence)
            det_list.append([x1, y1, x2, y2, score])
            cls_list.append(int(d.ClassID))
        det_arr = np.array(det_list, dtype=np.float32) if len(det_list) else np.zeros((0,5), np.float32)
        tracks = tracker.update(det_arr, class_ids=np.array(cls_list) if len(cls_list) else None)
        # 3) 트랙 ID/박스 오버레이
        # (detectNet가 박스 그리기까지는 해주지만, 트랙 ID 텍스트는 우리가 추가)
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            jetson_utils.cudaDrawRect(img, (int(x1), int(y1), int(x2), int(y2)), (0,255,0,255))
            font.OverlayText(img, w, h, f"id:{t.id} s:{t.score:.2f}",
                             int(x1), max(0, int(y1)-18),
                             font.White, font.Gray40)
        # 4) “2초 연속 감지”를 트랙 단위로 판단
        #   - 활성 트랙이 하나라도 있으면 is_present=True 로 해석
        now = time.time()
        is_present = len(tracks) > 0
        if is_present:
            if present_since is None:
                present_since = now
            elif (now - present_since) >= 2.0 and not audio_playing:
                play_mp3_once()
                audio_started = time.time()
                audio_playing = True
        else:
            present_since = None
        # 5) 오디오 재생 종료
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
            print("Timeout exit"); break
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
