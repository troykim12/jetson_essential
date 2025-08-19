#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultralytics YOLO + ByteTrack + Geofence Crossing Signal
- coords.json의 ROI_POLY / GEOFENCE_LINE을 불러와 오버레이
- YOLO(person만) + ByteTrack으로 ID를 유지
- 각 바운딩박스의 '중심점'이 지오펜스 선을 교차하면 IN/OUT 신호 발생

키:
  q : 종료
  l : coords.json 재로딩
  f : OUT/IN 기준 뒤집기(부호 반전 토글)
  o : 현재 프레임에서 '화면 중앙'을 OUT 기준으로 설정(초기 셋업용)

coords.json 예시:
{
  "ROI_POLY": [[120,350],[600,360],[610,510],[130,500]],
  "ROI_CLOSED": true,
  "GEOFENCE_LINE": [[100,400],[620,400]]
}
"""
import cv2, json, os, time
import numpy as np
from ultralytics import YOLO

# -------------------- 설정 --------------------
SAVE_PATH = r"C:\Users\USER\OneDrive\Desktop\calibration_images\coords.json"   # 필요시 절대경로로 변경
CAMERA_INDEX = 0            # 카메라 인덱스(환경에 맞게 0/1 등)
MODEL_PATH = "yolov8s.pt"   # 가벼운 모델 권장
CONF_THR = 0.4              # 신뢰도 임계값
CLS_PERSON = [0]            # COCO person 클래스만 추적

# -------------------- 좌표 로딩 --------------------
def load_coords(path=SAVE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poly = [tuple(p) for p in data.get("ROI_POLY", [])]
    closed = bool(data.get("ROI_CLOSED", True))  # 없으면 닫힌 것으로 처리
    geo = [tuple(p) for p in data.get("GEOFENCE_LINE", [])]
    if len(geo) != 2:
        raise ValueError("GEOFENCE_LINE must have exactly 2 points")
    return poly, closed, geo

# -------------------- 기하 유틸 --------------------
def side_of_line(pt, line):  # >0/<0 부호로 선의 한쪽/다른쪽
    (x1, y1), (x2, y2) = line
    x, y = pt
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def crossed(prev_pt, cur_pt, line):
    s1 = side_of_line(prev_pt, line)
    s2 = side_of_line(cur_pt,  line)
    return (s1 * s2) < 0 and abs(s1) > 1e-6 and abs(s2) > 1e-6

def draw_overlay(frame, poly_pts, poly_closed, geo_pts):
    disp = frame

    # ROI
    if poly_pts:
        pts_np = np.array(poly_pts, np.int32)
        if len(poly_pts) >= 2:
            cv2.polylines(disp, [pts_np], poly_closed or len(poly_pts) > 2, (0,200,255), 2, cv2.LINE_AA)
        if (poly_closed or len(poly_pts) > 2) and len(poly_pts) >= 3:
            mask = np.zeros_like(disp)
            cv2.fillPoly(mask, [pts_np], (0,200,255))
            disp = cv2.addWeighted(disp, 1.0, (mask*0.20).astype(disp.dtype), 0.20, 0)

    # GEOFENCE
    cv2.line(disp, geo_pts[0], geo_pts[1], (0,255,0), 3, cv2.LINE_AA)
    for p in geo_pts:
        cv2.circle(disp, p, 5, (0,255,0), -1)
    cv2.putText(disp, "GEOFENCE", (geo_pts[0][0]+10, geo_pts[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return disp

# -------------------- 신호 콜백 --------------------
def on_cross_in(tid, cx, cy):
    # TODO: 여기서 MQTT/HTTP/릴레이 등 원하는 신호를 붙이세요.
    print(f"[IN ] id={tid} at ({cx},{cy}) OUT->IN crossing")

def on_cross_out(tid, cx, cy):
    print(f"[OUT] id={tid} at ({cx},{cy}) IN->OUT crossing")

# -------------------- 메인 --------------------
def main():
    poly_pts, poly_closed, geo_pts = load_coords()
    print("[INFO] ROI_POLY pts:", len(poly_pts), "closed:", poly_closed)
    print("[INFO] GEOFENCE_LINE:", geo_pts)

    # OUT 기준(부호)
    OUT_SIGN = -1  # side_of_line의 부호가 이 값이면 OUT, 반대면 IN

    # 이전 상태 저장
    prev_pos = {}     # tid -> (cx,cy)
    debounce = {}     # tid -> last event time
    DEBOUNCE_SEC = 0.5

    # YOLO + ByteTrack
    model = YOLO(MODEL_PATH)

    # Ultralytics track 제너레이터 사용(프레임 단위 결과)
    stream = model.track(
        source=CAMERA_INDEX,
        stream=True,
        verbose=False,
        tracker="bytetrack.yaml",   # 기본 ByteTrack
        persist=True,               # ID 유지
        classes=CLS_PERSON,         # person만
        conf=CONF_THR,
        show=False
    )

    WIN = "YOLO Geofence Crossing"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("[HELP] q:quit  l:reload json  f:flip OUT/IN  o:set center as OUT")
    for res in stream:
        frame = res.orig_img
        if frame is None:
            continue

        disp = frame.copy()
        disp = draw_overlay(disp, poly_pts, poly_closed, geo_pts)

        # 결과 박스
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            ids   = boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else np.array([-1]*len(xyxy))

            for bbox, tid in zip(xyxy, ids):
                x1,y1,x2,y2 = bbox.astype(int)
                cx, cy = int((x1+x2)/2), int(y2)

                # 시각화
                cv2.rectangle(disp, (x1,y1), (x2,y2), (255,255,0), 2)
                cv2.circle(disp, (cx,cy), 5, (50,180,255), -1)             # 발 위치
                cv2.line(disp, (x1, y2), (x2, y2), (255,255,0), 1)   
                cv2.putText(disp, f"id={tid}", (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(disp, f"id={tid}", (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

                # 최초 저장
                if tid not in prev_pos:
                    prev_pos[tid] = (cx, cy)
                    continue

                # 교차 판정
                p = prev_pos[tid]
                if crossed(p, (cx,cy), geo_pts):
                    s_prev = 1 if side_of_line(p,       geo_pts) >= 0 else -1
                    s_cur  = 1 if side_of_line((cx,cy), geo_pts) >= 0 else -1

                    was_out = (s_prev == OUT_SIGN)
                    now_in  = (s_cur  == -OUT_SIGN)

                    tnow = time.time()
                    if tnow - debounce.get(tid, 0) > DEBOUNCE_SEC:
                        if was_out and now_in:
                            on_cross_in(tid, cx, cy)
                            cv2.putText(disp, "OUT->IN", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(disp, "OUT->IN", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                        else:
                            on_cross_out(tid, cx, cy)
                            cv2.putText(disp, "IN->OUT", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(disp, "IN->OUT", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2, cv2.LINE_AA)
                        debounce[tid] = tnow

                prev_pos[tid] = (cx, cy)

        # HUD
        hud = "q:quit  l:reload json  f:flip OUT/IN  o:set center as OUT"
        cv2.putText(disp, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('l'):
            try:
                poly_pts, poly_closed, geo_pts = load_coords()
                print("[RELOAD] coords.json reloaded.")
            except Exception as e:
                print("[RELOAD ERROR]", e)
        elif k == ord('f'):
            OUT_SIGN *= -1
            print("[FLIP] OUT_SIGN:", OUT_SIGN)
        elif k == ord('o'):
            # 화면 중앙 픽셀을 OUT으로 지정(초기 셋업용)
            h, w = disp.shape[:2]
            s_now = side_of_line((w//2, h//2), geo_pts)
            OUT_SIGN = 1 if s_now >= 0 else -1
            print("[SET OUT] screen center OUT_SIGN =", OUT_SIGN)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
