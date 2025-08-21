#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8(person) + ByteTrack
+ Geofence Crossing (IN/OUT)
+ Loitering (speed-aware + exit grace)
+ ID Stabilization (IoU remap)
+ Hand Gestures (OK -> ThumbsUp) to enroll OWNER (exempt from loiter/intrusion)
+ Interactive ROI/LOITER/GEOFENCE editor + HUD

Tip:
- If mediapipe install fails on your Python, the code falls back to cvzone.
- If both are unavailable, gesture enrollment is disabled but other features still run.
"""

import cv2, json, os, time
import numpy as np
from ultralytics import YOLO

# ==================== CONFIG ====================
SAVE_PATH = r"C:\Users\USER\OneDrive\Desktop\calibration_images\coords.json"
CAMERA_INDEX = 0
MODEL_PATH = "yolov8s.pt"
CONF_THR = 0.4
CLS_PERSON = [0]
DEBOUNCE_SEC = 1.0
LOITER_SEC = 5.0
TRACKER_CFG = "bytetrack.yaml"

# ID stabilization / speed-based loitering / exit grace
IOU_REMAP_THR = 0.6      # IoU threshold for id remap
REMAP_T_MAX   = 0.7      # seconds to allow remap after lost
SPEED_THR     = 30.0     # px/s; only count dwell when slower than this
SPEED_EMA_A   = 0.6      # EMA alpha for speed
EXIT_GRACE_SEC = 0.8     # allow brief outside (seconds) without resetting dwell

# Gesture enrollment params
OK_HOLD_FRAMES   = 3     # consecutive frames for OK
GOOD_HOLD_FRAMES = 3     # consecutive frames for ThumbsUp
ENROLL_WINDOW_SEC = 10.0  # OK must be followed by GOOD within this window
HAND_DETECT_CONF = 0.5
HAND_TRACK_CONF  = 0.45

# ==================== HAND BACKEND (mediapipe / cvzone / none) ====================
HAND_BACKEND = "none"
mp = None
CVZHandDetector = None
try:
    import mediapipe as _mp
    mp = _mp
    HAND_BACKEND = "mediapipe"
except Exception:
    try:
        from cvzone.HandTrackingModule import HandDetector as _CVZHandDetector
        CVZHandDetector = _CVZHandDetector
        HAND_BACKEND = "cvzone"
    except Exception:
        HAND_BACKEND = "none"

# ==================== GEOMETRY / UTILS ====================
def load_coords(path=SAVE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roi = [tuple(p) for p in data.get("ROI_POLY", [])]
    loiter = [tuple(p) for p in data.get("LOITER_POLY", [])]
    roi_closed = bool(data.get("ROI_CLOSED", True))
    geo = [tuple(p) for p in data.get("GEOFENCE_LINE", [])]
    if len(geo) != 2:
        raise ValueError("GEOFENCE_LINE must have exactly 2 points")
    return roi, roi_closed, geo, loiter

def side_of_line(pt, line):
    (x1, y1), (x2, y2) = line; x, y = pt
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def crossed(prev_pt, cur_pt, line):
    s1 = side_of_line(prev_pt, line); s2 = side_of_line(cur_pt, line)
    return (s1 * s2) < 0 and abs(s1) > 1e-6 and abs(s2) > 1e-6

def point_in_poly(pt, poly_pts):
    if not poly_pts or len(poly_pts) < 3: return False
    x, y = pt; inside = False; n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]; x2, y2 = poly_pts[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        if cond: inside = not inside
    return inside

def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw*ih
    area_a = max(0, xa2-xa1)*max(0, ya2-ya1)
    area_b = max(0, xb2-xb1)*max(0, yb2-yb1)
    union = area_a + area_b - inter + 1e-9
    return inter/union

# ==================== DRAW ====================
def draw_overlay(frame, roi_pts, roi_closed, geo_pts, loiter_pts):
    disp = frame
    # ROI
    if roi_pts:
        rnp = np.array(roi_pts, np.int32)
        cv2.polylines(disp, [rnp], roi_closed or len(roi_pts) > 2, (0,200,255), 2, cv2.LINE_AA)
        if (roi_closed or len(roi_pts) > 2) and len(roi_pts) >= 3:
            mask = np.zeros_like(disp); cv2.fillPoly(mask, [rnp], (0,200,255))
            disp = cv2.addWeighted(disp, 1.0, (mask*0.12).astype(disp.dtype), 0.12, 0)
    # LOITER
    if loiter_pts and len(loiter_pts) >= 3:
        lnp = np.array(loiter_pts, np.int32)
        cv2.polylines(disp, [lnp], True, (0,0,255), 2, cv2.LINE_AA)
        lmask = np.zeros_like(disp); cv2.fillPoly(lmask, [lnp], (0,0,255))
        disp = cv2.addWeighted(disp, 1.0, (lmask*0.10).astype(disp.dtype), 0.10, 0)
        (lx, ly) = lnp.mean(axis=0).astype(int)
        cv2.putText(disp, "LOITER AREA", (lx-60, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, "LOITER AREA", (lx-60, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    # GEOFENCE
    if len(geo_pts) == 2:
        cv2.line(disp, geo_pts[0], geo_pts[1], (0,255,0), 3, cv2.LINE_AA)
        for p in geo_pts: cv2.circle(disp, p, 5, (0,255,0), -1)
        cv2.putText(disp, "GEOFENCE", (geo_pts[0][0]+10, geo_pts[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return disp

# ==================== EVENTS ====================
def on_cross_in(tid, cx, cy):
    print(f"[CROSS IN] id={tid} @({cx},{cy})")

def on_cross_out(tid, cx, cy):
    print(f"[CROSS OUT] id={tid} @({cx},{cy})")

def on_loitering(tid, dwell_sec, cx, cy):
    print(f"[LOITER] id={tid}, dwell={dwell_sec:.1f}s @({cx},{cy})")

# ==================== INTERACTIVE EDITOR ====================
class InteractiveEditor:
    def __init__(self, win_name):
        self.win = win_name
        self.mode = None  # 'roi' | 'loiter' | 'geo'
        self.roi_pts = []
        self.roi_closed = True
        self.loiter_pts = []
        self.geo_pts = [(0,0), (100,0)]
        self.drag_idx = (-1)
        self.radius = 12
        cv2.setMouseCallback(self.win, self._on_mouse)

    def set_from_loaded(self, roi_pts, roi_closed, geo_pts, loiter_pts):
        self.roi_pts = [(int(x), int(y)) for (x,y) in roi_pts]
        self.roi_closed = bool(roi_closed)
        self.geo_pts = [(int(geo_pts[0][0]), int(geo_pts[0][1])),
                        (int(geo_pts[1][0]), int(geo_pts[1][1]))]
        self.loiter_pts = [(int(x), int(y)) for (x,y) in loiter_pts]

    def _nearest_idx(self, pts, x, y):
        if not pts: return -1, 1e9
        dists = [ (i, (px-x)**2 + (py-y)**2) for i,(px,py) in enumerate(pts) ]
        idx, d2 = min(dists, key=lambda t:t[1])
        return (idx, np.sqrt(d2))

    def _on_mouse(self, event, x, y, flags, param):
        if self.mode is None: return
        if self.mode == 'geo':
            idx0, d0 = self._nearest_idx(self.geo_pts, x, y)
            if event == cv2.EVENT_LBUTTONDOWN and d0 <= self.radius*2:
                self.drag_idx = ('geo', idx0)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                if self.drag_idx and self.drag_idx[0]=='geo':
                    gi = self.drag_idx[1]
                    self.geo_pts[gi] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drag_idx = (-1)
            return

        pts = self.roi_pts if self.mode=='roi' else self.loiter_pts
        idx, d = self._nearest_idx(pts, x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if idx != -1 and d <= self.radius:
                self.drag_idx = (self.mode, idx)
            else:
                pts.append((x,y))
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.drag_idx and self.drag_idx[0] == self.mode and self.drag_idx[1] != -1:
                pts[self.drag_idx[1]] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = (-1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if idx != -1 and d <= self.radius:
                pts.pop(idx)

        if self.mode=='roi': self.roi_pts = pts
        else: self.loiter_pts = pts

    def handle_key(self, k):
        if k == ord('r'):
            self.mode = 'roi'; print("[EDIT] ROI mode")
        elif k == ord('k'):
            self.mode = 'loiter'; print("[EDIT] LOITER mode")
        elif k == ord('g'):
            self.mode = 'geo'; print("[EDIT] GEOFENCE mode")
        elif k == ord('t'):
            if self.mode == 'roi':
                self.roi_closed = not self.roi_closed
                print("[TOGGLE] ROI_CLOSED =", self.roi_closed)
            elif self.mode == 'loiter':
                print("[INFO] LOITER poly is always closed (visual only).")
        elif k == ord('c'):
            if self.mode == 'roi':
                self.roi_pts = []; print("[CLEAR] ROI cleared")
            elif self.mode == 'loiter':
                self.loiter_pts = []; print("[CLEAR] LOITER cleared")
            elif self.mode == 'geo':
                self.geo_pts = [(0,0),(100,0)]; print("[CLEAR] GEOFENCE reset")
        elif k == 27:
            self.mode = None; print("[EDIT] exit edit mode")

    def draw_handles(self, img):
        color = {'roi':(0,200,255), 'loiter':(0,0,255)}
        for p in self.roi_pts:
            cv2.circle(img, p, 5, color['roi'], -1)
        for p in self.loiter_pts:
            cv2.circle(img, p, 5, color['loiter'], -1)
        for i,p in enumerate(self.geo_pts):
            cv2.circle(img, p, 7, (0,255,0) if i==0 else (0,180,0), -1)

# ==================== HAND LANDMARKS (unified) ====================
# Indices (MediaPipe Hands convention):
WRIST=0; THUMB_CMC=1; THUMB_MCP=2; THUMB_IP=3; THUMB_TIP=4
INDEX_MCP=5; INDEX_PIP=6; INDEX_DIP=7; INDEX_TIP=8
MIDDLE_MCP=9; MIDDLE_PIP=10; MIDDLE_DIP=11; MIDDLE_TIP=12
RING_MCP=13; RING_PIP=14; RING_DIP=15; RING_TIP=16
PINKY_MCP=17; PINKY_PIP=18; PINKY_DIP=19; PINKY_TIP=20

def _dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def hand_center_from_px(px21):
    xs = [p[0] for p in px21]; ys = [p[1] for p in px21]
    return (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))

def is_ok_sign_px(px21):
    # Thumb-Index circle + other three extended (relaxed)
    tip_thumb = px21[THUMB_TIP]
    tip_index = px21[INDEX_TIP]
    tip_mid   = px21[MIDDLE_TIP]; pip_mid  = px21[MIDDLE_PIP]
    tip_ring  = px21[RING_TIP];   pip_ring = px21[RING_PIP]
    tip_pink  = px21[PINKY_TIP];  pip_pink = px21[PINKY_PIP]
    wrist     = px21[WRIST]

    # 손 스케일(대략): 손목-중지 끝 거리
    scale = _dist(wrist, tip_mid) + 1e-6
    tol   = 0.06 * scale           # ← 여유 (픽셀 기준)

    # 동그라미 크기: 조금 더 허용
    circle_d  = _dist(tip_thumb, tip_index)
    circle_ok = (circle_d / scale) < 0.35     # ← 0.25 → 0.35 로 완화

    # '펴짐' 판정 완화: tip이 pip보다 약간만 위여도 OK
    mid_up  = tip_mid[1]  < (pip_mid[1]  - tol*0.3)
    ring_up = tip_ring[1] < (pip_ring[1] - tol*0.3)
    pink_up = tip_pink[1] < (pip_pink[1] - tol*0.3)

    return circle_ok and (mid_up and ring_up and pink_up)


def is_thumbs_up_px(px21):
    tip_thumb = px21[THUMB_TIP];  ip_thumb  = px21[THUMB_IP];  mcp_thumb = px21[THUMB_MCP]
    tip_index = px21[INDEX_TIP];  pip_index = px21[INDEX_PIP]
    tip_mid   = px21[MIDDLE_TIP]; pip_mid   = px21[MIDDLE_PIP]
    tip_ring  = px21[RING_TIP];   pip_ring  = px21[RING_PIP]
    tip_pink  = px21[PINKY_TIP];  pip_pink  = px21[PINKY_PIP]
    wrist     = px21[WRIST]

    # 손 스케일 기반 허용 오차
    scale = _dist(wrist, tip_mid) + 1e-6
    tol   = 0.06 * scale

    # 엄지 위로(완화): tip < ip < mcp 를 유지하되, 약간의 오차 허용
    thumb_up = (tip_thumb[1] < ip_thumb[1] - tol*0.2) and (ip_thumb[1] < mcp_thumb[1] - tol*0.2)

    # 나머지 손가락 접힘(완화): tip 이 pip 보다 '조금'만 아래여도 인정
    idx_fold  = tip_index[1] > (pip_index[1] - tol*0.2)
    mid_fold  = tip_mid[1]   > (pip_mid[1]  - tol*0.2)
    ring_fold = tip_ring[1]  > (pip_ring[1] - tol*0.2)
    pink_fold = tip_pink[1]  > (pip_pink[1] - tol*0.2)

    return thumb_up and idx_fold and mid_fold and ring_fold and pink_fold


class HandBridge:
    """Unifies mediapipe and cvzone outputs into list of 21 (x,y) pixel landmarks."""
    def __init__(self):
        self.backend = HAND_BACKEND
        self._mp_hands = None
        self._cvz = None
        if self.backend == "mediapipe":
            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=HAND_DETECT_CONF,
                min_tracking_confidence=HAND_TRACK_CONF
            )
        elif self.backend == "cvzone":
            self._cvz = CVZHandDetector(detectionCon=HAND_DETECT_CONF, maxHands=2)

    def get_hands_px(self, frame_bgr):
        """Return list of hands; each is list of 21 (x,y) pixel coords."""
        h, w = frame_bgr.shape[:2]
        out = []
        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._mp_hands.process(rgb)
            if res.multi_hand_landmarks:
                for hlm in res.multi_hand_landmarks:
                    px21 = []
                    for lm in hlm.landmark:
                        px21.append((int(lm.x * w), int(lm.y * h)))
                    if len(px21) == 21:
                        out.append(px21)
        elif self.backend == "cvzone":
            hands, _ = self._cvz.findHands(frame_bgr, draw=False, flipType=False)
            if hands:
                for hand in hands:
                    lm = hand.get('lmList', None)
                    if lm and len(lm) >= 21:
                        # lm entries are [x, y, z]
                        px21 = [(int(p[0]), int(p[1])) for p in lm[:21]]
                        out.append(px21)
        else:
            # no backend
            pass
        return out

    def close(self):
        if self.backend == "mediapipe" and self._mp_hands:
            self._mp_hands.close()

# ==================== MAIN ====================
def main():
    roi_pts, roi_closed, geo_pts, loiter_pts = load_coords()

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): raise RuntimeError("Camera open failed")

    WIN = "YOLO+ByteTrack Geofence+Loitering (Interactive+StableID+Speed+Gestures)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    editor = InteractiveEditor(WIN)
    editor.set_from_loaded(roi_pts, roi_closed, geo_pts, loiter_pts)

    # Hand bridge
    hand_bridge = HandBridge()
    gesture_enabled = (hand_bridge.backend != "none")
    if not gesture_enabled:
        print("[WARN] Gesture enrollment disabled (mediapipe/cvzone not available).")

    # ---- States ----
    OUT_SIGN = 1
    loiter_total = 0

    prev_pos = {}        # sid -> (cx,cy) for crossing
    last_time = {}       # sid -> last timestamp (for dt)
    speed_ema = {}       # sid -> EMA(px/s)

    dwell_accum = {}     # sid -> dwell seconds (only when slow & inside)
    loiter_flag = {}     # sid -> loiter fired
    exit_time = {}       # sid -> last time exited loiter area (for grace)

    remap = {}           # tid -> sid
    last_bbox = {}       # sid -> last bbox (x1,y1,x2,y2)
    lost_cache = {}      # sid -> {'bbox':..., 't': last_seen_time}

    # Gesture/Owner
    owner_ids = set()    # enrolled owners (sid)
    ok_count  = {}       # sid -> consecutive OK frames
    good_count= {}       # sid -> consecutive GOOD frames
    ok_time   = {}       # sid -> last OK fulfilled time

    while True:
        ok, frame = cap.read()
        if not ok: break

        cur_roi = editor.roi_pts
        cur_roi_closed = editor.roi_closed
        cur_loiter = editor.loiter_pts if len(editor.loiter_pts) >= 3 else cur_roi
        cur_geo = editor.geo_pts

        results = model.track(source=frame, stream=False, persist=True,
                              conf=CONF_THR, classes=CLS_PERSON,
                              tracker=TRACKER_CFG, verbose=False)

        if len(results) == 0:
            disp = draw_overlay(frame.copy(), cur_roi, cur_roi_closed, cur_geo, cur_loiter)
            editor.draw_handles(disp)
            cv2.putText(disp, "(No detections)", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, "(No detections)", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if   k == ord('q'): break
            elif k == ord('l'):
                try:
                    roi_pts, roi_closed, geo_pts, loiter_pts = load_coords()
                    editor.set_from_loaded(roi_pts, roi_closed, geo_pts, loiter_pts)
                    print("[RELOAD] coords.json reloaded.")
                except Exception as e:
                    print("[RELOAD ERROR]", e)
            elif k == ord('f'):
                OUT_SIGN = -1 if OUT_SIGN==1 else 1
                print("[FLIP] OUT_SIGN:", OUT_SIGN)
            elif k == ord('o'):
                h, w = disp.shape[:2]
                s_now = side_of_line((w//2, h//2), editor.geo_pts)
                OUT_SIGN = 1 if s_now >= 0 else -1
                print("[SET OUT] screen center OUT_SIGN =", OUT_SIGN)
            else:
                editor.handle_key(k)
            continue

        res = results[0]
        disp = draw_overlay(res.orig_img.copy(), cur_roi, cur_roi_closed, cur_geo, cur_loiter)
        editor.draw_handles(disp)

        boxes = getattr(res, "boxes", None)
        tnow = time.time()
        seen_stable = set()
        seen_current = set()

        # ---- (1) current tids & bboxes
        curr = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            tids = boxes.id
            tids = tids.cpu().numpy().astype(int) if tids is not None else np.array([-1]*len(xyxy))
            for bbox, tid in zip(xyxy, tids):
                seen_current.add(tid)
                x1,y1,x2,y2 = bbox.astype(int)
                curr.append((tid, (x1,y1,x2,y2)))

        # ---- (2) ID remap using lost_cache (IoU)
        for tid, bbox in curr:
            if tid in remap:  # already remapped
                continue
            best_sid, best_iou = None, 0.0
            for sid, info in list(lost_cache.items()):
                dt_lost = tnow - info['t']
                if dt_lost > REMAP_T_MAX:
                    continue
                iou = iou_xyxy(bbox, info['bbox'])
                if iou > IOU_REMAP_THR and iou > best_iou:
                    best_sid, best_iou = sid, iou
            if best_sid is not None:
                remap[tid] = best_sid

        # ---- (3) Gestures: detect hands, map to nearest/inside person, enroll OWNER
        if gesture_enabled and len(curr) > 0:
            hands_px = hand_bridge.get_hands_px(frame)
            if hands_px:
                # Prepare centers for persons
                person_centers = []
                for _, b in curr:
                    x1,y1,x2,y2 = b
                    person_centers.append( ((x1+x2)//2, (y1+y2)//2) )

                for px21 in hands_px:
                    hc = hand_center_from_px(px21)
                    # find best person: inside bbox wins; else nearest center
                    best_tid, best_d = None, 1e9
                    for i, (tid, b) in enumerate(curr):
                        x1,y1,x2,y2 = b
                        if x1 <= hc[0] <= x2 and y1 <= hc[1] <= y2:
                            best_tid, best_d = tid, 0.0
                            break
                        d = _dist(hc, person_centers[i])
                        if d < best_d:
                            best_tid, best_d = tid, d
                    if best_tid is None:
                        continue
                    sid = remap.get(best_tid, best_tid)

                    ok_now = is_ok_sign_px(px21)
                    good_now = is_thumbs_up_px(px21)

                    # 기존:
                    # ok_count[sid]   = ok_count.get(sid, 0)   + (1 if ok_now else -ok_count.get(sid, 0))
                    # good_count[sid] = good_count.get(sid, 0) + (1 if good_now else -good_count.get(sid, 0))

                    # 완화(느린 감쇠):
                    ok_prev   = ok_count.get(sid, 0)
                    good_prev = good_count.get(sid, 0)
                    ok_count[sid]   = ok_prev   + 1 if ok_now   else max(0, ok_prev   - 1)
                    good_count[sid] = good_prev + 1 if good_now else max(0, good_prev - 1)


                    if ok_count[sid] >= OK_HOLD_FRAMES:
                        ok_time[sid] = tnow
                        cv2.putText(disp, "OK✔", (hc[0]+6, hc[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(disp, "OK✔", (hc[0]+6, hc[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

                    if good_count[sid] >= GOOD_HOLD_FRAMES:
                        t_ok = ok_time.get(sid, 0)
                        if t_ok and (tnow - t_ok) <= ENROLL_WINDOW_SEC:
                            owner_ids.add(sid)
                            cv2.putText(disp, "OWNER ENROLLED", (hc[0]-60, hc[1]-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(disp, "OWNER ENROLLED", (hc[0]-60, hc[1]-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)
                            ok_count[sid] = 0; good_count[sid] = 0; ok_time.pop(sid, None)

                    cv2.circle(disp, hc, 6, (0,200,0), -1)

        # ---- (4) Main tracking loop: crossing, speed, loiter
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            tids = boxes.id
            tids = tids.cpu().numpy().astype(int) if tids is not None else np.array([-1]*len(xyxy))

            for bbox, tid in zip(xyxy, tids):
                x1,y1,x2,y2 = bbox.astype(int)
                cx, cy = int((x1+x2)/2), int(y2)

                sid = remap.get(tid, tid)
                seen_stable.add(sid)

                # draw bbox/id
                cv2.rectangle(disp, (x1,y1), (x2,y2), (255,255,0), 2)
                cv2.circle(disp, (cx,cy), 5, (50,180,255), -1)
                cv2.line(disp, (x1, y2), (x2, y2), (255,255,0), 1)
                id_text = f"id={sid}" if sid==tid else f"id={sid}({tid})"
                cv2.putText(disp, id_text, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(disp, id_text, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

                # --- OWNER exemption ---
                if sid in owner_ids:
                    cv2.putText(disp, "OWNER", (x1, y1-28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(disp, "OWNER", (x1, y1-28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2, cv2.LINE_AA)
                    # keep state updates minimal
                    prev_pos[sid] = (cx, cy)
                    last_bbox[sid] = (x1,y1,x2,y2)
                    last_time[sid] = tnow
                    dwell_accum[sid] = 0.0
                    loiter_flag.pop(sid, None)
                    # Skip crossing/loiter logic
                    continue

                # Geofence crossing using foot point
                if sid not in prev_pos:
                    prev_pos[sid] = (cx, cy)
                else:
                    p = prev_pos[sid]
                    if crossed(p, (cx,cy), cur_geo):
                        s_prev = 1 if side_of_line(p, cur_geo) >= 0 else -1
                        s_cur  = 1 if side_of_line((cx,cy), cur_geo) >= 0 else -1
                        was_out = (s_prev == OUT_SIGN); now_in = (s_cur == -OUT_SIGN)
                        last_ev = last_time.get(('ev', sid), 0.0)
                        if tnow - last_ev > DEBOUNCE_SEC:
                            if was_out and now_in:
                                on_cross_in(sid, cx, cy)
                                cv2.putText(disp, "OUT->IN", (30, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                                cv2.putText(disp, "OUT->IN", (30, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                            else:
                                on_cross_out(sid, cx, cy)
                                cv2.putText(disp, "IN->OUT", (30, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                                cv2.putText(disp, "IN->OUT", (30, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2, cv2.LINE_AA)
                            last_time[('ev', sid)] = tnow
                    prev_pos[sid] = (cx, cy)

                # Speed estimate (EMA)
                dt = None
                if sid not in last_time:
                    last_time[sid] = tnow
                else:
                    dt = max(1e-3, tnow - last_time[sid])
                    last_time[sid] = tnow

                if dt is not None and sid in last_bbox:
                    lx1,ly1,lx2,ly2 = last_bbox[sid]
                    pcx, pcy = int((lx1+lx2)/2), int(ly2)
                    dist = ((cx - pcx)**2 + (cy - pcy)**2) ** 0.5
                    v = dist / dt
                    speed_ema[sid] = v if sid not in speed_ema else (SPEED_EMA_A * v + (1.0 - SPEED_EMA_A) * speed_ema[sid])
                else:
                    speed_ema.setdefault(sid, 0.0)
                last_bbox[sid] = (x1,y1,x2,y2)

                # Loitering (speed-aware + exit grace)
                inside_loiter = point_in_poly((cx,cy), cur_loiter) if (cur_loiter and len(cur_loiter)>=3) else False
                v_now = speed_ema.get(sid, 0.0)
                if sid not in dwell_accum: dwell_accum[sid] = 0.0

                if inside_loiter:
                    if sid in exit_time:
                        if (tnow - exit_time[sid]) <= EXIT_GRACE_SEC:
                            pass  # keep dwell_accum
                        else:
                            dwell_accum[sid] = 0.0
                            loiter_flag.pop(sid, None)
                        exit_time.pop(sid, None)
                    if dt is not None and v_now < SPEED_THR:
                        dwell_accum[sid] += dt

                    cv2.putText(disp, f"dwell {dwell_accum[sid]:.1f}s  v={v_now:.1f}px/s", (x1, y2+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(disp, f"dwell {dwell_accum[sid]:.1f}s  v={v_now:.1f}px/s", (x1, y2+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

                    if (dwell_accum[sid] >= LOITER_SEC) and not loiter_flag.get(sid, False):
                        on_loitering(sid, dwell_accum[sid], cx, cy)
                        loiter_flag[sid] = True
                        loiter_total += 1
                        cv2.putText(disp, "LOITERING", (x1, y1-28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(disp, "LOITERING", (x1, y1-28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 3)
                else:
                    if sid not in exit_time:
                        exit_time[sid] = tnow
                    else:
                        if (tnow - exit_time[sid]) > EXIT_GRACE_SEC:
                            dwell_accum[sid] = 0.0
                            loiter_flag.pop(sid, None)
                            exit_time.pop(sid, None)

        # ---- (5) lost_cache maintenance for remap
        all_known = set(list(prev_pos.keys()) + list(last_bbox.keys()) + list(dwell_accum.keys()))
        for sid in list(all_known):
            if sid not in seen_stable:
                if sid in last_bbox:
                    lost_cache[sid] = {'bbox': last_bbox[sid], 't': tnow}
            else:
                lost_cache.pop(sid, None)
        for sid in list(lost_cache.keys()):
            if tnow - lost_cache[sid]['t'] > REMAP_T_MAX:
                lost_cache.pop(sid, None)

        # ---- HUD ----
        top = f"MODE:[{'None' if editor.mode is None else editor.mode.upper()}]  ROI_CLOSED:{editor.roi_closed}  OUT_SIGN:{'+' if OUT_SIGN==1 else '-'}  LOITER_TOTAL:{loiter_total}  OWNERS:{len(owner_ids)}"
        cv2.putText(disp, top, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, top, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)

        help1 = "r:edit ROI  k:edit LOITER  g:edit GEOFENCE  t:toggle close  c:clear  l:reload json"
        help2 = "o:set center as OUT  f:flip OUT/IN  q:quit  (mouse: L-add/move/drag, R-delete)"
        cv2.putText(disp, help1, (12, disp.shape[0]-36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, help1, (12, disp.shape[0]-36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(disp, help2, (12, disp.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, help2, (12, disp.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if   k == ord('q'):
            break
        elif k == ord('l'):
            try:
                roi_pts, roi_closed, geo_pts, loiter_pts = load_coords()
                editor.set_from_loaded(roi_pts, roi_closed, geo_pts, loiter_pts)
                print("[RELOAD] coords.json reloaded.")
            except Exception as e:
                print("[RELOAD ERROR]", e)
        elif k == ord('f'):
            OUT_SIGN = -1 if OUT_SIGN==1 else 1
            print("[FLIP] OUT_SIGN:", OUT_SIGN)
        elif k == ord('o'):
            h, w = disp.shape[:2]
            s_now = side_of_line((w//2, h//2), editor.geo_pts)
            OUT_SIGN = 1 if s_now >= 0 else -1
            print("[SET OUT] screen center OUT_SIGN =", OUT_SIGN)
        else:
            editor.handle_key(k)

    cap.release()
    hand_bridge.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
