#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image ROI / Geofence Annotator (single image)
- 좌클릭 : 현재 모드의 점 추가
- 우클릭 : 현재 모드의 마지막 점 삭제
- m : 모드 전환 (POLY <-> GEOFENCE)
- c : 폴리곤 닫기/열기 토글
- g : 지오펜스 선 확정(2점)
- r : 현재 모드 좌표 리셋
- s : coords.json 저장 (ROI_POLY, GEOFENCE_LINE)
- l : coords.json 불러오기
- q : 종료
"""
#주의: 이 코드는 단일 이미지에 대한 주석 작업을 위한 것입니다. 실시간 카메라 스트림이 아닙니다.

import cv2, json, os
import numpy as np

WIN = "Annotator"
SAVE_PATH = "coords.json" #좌표가 jason형태로 저장될 젯슨 나노의 경로로 변경 필요함.
IMG_PATH = r"C:\Users\USER\OneDrive\Desktop\calibration_images\capture_gst.jpg"  # ← 이미지가 저장되어 있는 젯슨 나노의 경로가 필요함.

mode = "POLY"          # 또는 "GEOFENCE"
poly_pts = []          # [(x,y), ...]
poly_closed = False
geo_pts  = []          # 2점만 허용
geo_fixed = False

def load_coords():
    global poly_pts, poly_closed, geo_pts, geo_fixed
    if not os.path.exists(SAVE_PATH):
        return False
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    poly_pts   = [tuple(map(int, p)) for p in data.get("ROI_POLY", [])]
    poly_closed = bool(data.get("ROI_CLOSED", False))
    geo_pts    = [tuple(map(int, p)) for p in data.get("GEOFENCE_LINE", [])][:2]
    geo_fixed  = len(geo_pts) == 2
    print("[LOAD]", data)
    return True

def save_coords():
    data = {
        "ROI_POLY": poly_pts,
        "ROI_CLOSED": poly_closed,
        "GEOFENCE_LINE": geo_pts
    }
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[SAVE] ->", SAVE_PATH)

def on_mouse(event, x, y, flags, param):
    global mode, poly_pts, poly_closed, geo_pts, geo_fixed  # ✅ 함수 시작에 선언
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "POLY":
            if poly_closed:
                print("[INFO] 폴리곤이 닫혀 있어요. 'c'로 다시 열고 점을 추가하세요.")
            else:
                poly_pts.append((x, y))
                print(f"[POLY+] {x},{y}")
        else:  # GEOFENCE
            if geo_fixed:
                print("[INFO] 지오펜스가 확정됨. 'r'로 초기화하거나 'm'으로 모드 변경.")
            else:
                if len(geo_pts) < 2:
                    geo_pts.append((x, y))
                    print(f"[GEOFENCE+] {x},{y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode == "POLY":
            if poly_pts:
                print(f"[POLY-] remove {poly_pts[-1]}")
                poly_pts.pop()
        else:
            if geo_pts and not geo_fixed:
                print(f"[GEOFENCE-] remove {geo_pts[-1]}")
                geo_pts.pop()

def draw_overlay(base_img):
    disp = base_img.copy()

    # 안내 텍스트
    hud = [
        f"MODE: {mode} | 'm' switch, Left-Click add / Right-Click undo",
        "POLY: 'c' close/open, 'r' reset",
        "GEOFENCE: need 2 points, 'g' fix, 'r' reset",
        "'s' save, 'l' load, 'q' quit",
    ]
    y0 = 24
    for i, t in enumerate(hud):
        cv2.putText(disp, t, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, t, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # 폴리곤 그리기
    if poly_pts:
        color = (0, 200, 255)
        for i, p in enumerate(poly_pts):
            cv2.circle(disp, p, 4, (0, 255, 255), -1)
            cv2.putText(disp, str(i+1), (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, str(i+1), (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        pts_np = np.array(poly_pts, np.int32)
        if len(poly_pts) >= 2:
            cv2.polylines(disp, [pts_np], poly_closed, color, 2, cv2.LINE_AA)
        if poly_closed and len(poly_pts) >= 3:
            mask = np.zeros_like(disp)
            cv2.fillPoly(mask, [pts_np], (0, 200, 255))
            disp = cv2.addWeighted(disp, 1.0, (mask*0.3).astype(disp.dtype), 0.3, 0)

    # 지오펜스 선 그리기
    if geo_pts:
        c = (0, 255, 0) if geo_fixed else (180, 255, 100)
        for p in geo_pts:
            cv2.circle(disp, p, 5, c, -1)
        if len(geo_pts) == 2:
            cv2.line(disp, geo_pts[0], geo_pts[1], c, 3, cv2.LINE_AA)
            cv2.putText(
                disp, "GEOFENCE (OUT->IN choose later in code)",
                (min(geo_pts[0][0], geo_pts[1][0]) + 10, min(geo_pts[0][1], geo_pts[1][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA
            )

    return disp

def main():
    global mode, poly_pts, poly_closed, geo_pts, geo_fixed  # ✅ 함수 시작에 선언

    # 1) 이미지 로드 (웹캠 대체)
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {IMG_PATH}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    print("이미지 주석 모드… (q 종료)")
    while True:
        disp = draw_overlay(img)

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            mode = "GEOFENCE" if mode == "POLY" else "POLY"
            print("[MODE]", mode)
        elif k == ord('c') and mode == "POLY":
            if len(poly_pts) >= 3:
                poly_closed = not poly_closed
                print("[POLY] closed =", poly_closed)
            else:
                print("[WARN] 폴리곤은 3점 이상 필요")
        elif k == ord('g') and mode == "GEOFENCE":
            if len(geo_pts) == 2:
                geo_fixed = not geo_fixed
                print("[GEOFENCE] fixed =", geo_fixed)
            else:
                print("[WARN] 지오펜스는 2점 필요")
        elif k == ord('r'):
            if mode == "POLY":
                poly_pts.clear()
                poly_closed = False
                print("[RESET] POLY cleared")
            else:
                geo_pts.clear()
                geo_fixed = False
                print("[RESET] GEOFENCE cleared")
        elif k == ord('s'):
            save_coords()
        elif k == ord('l'):
            if not load_coords():
                print("[WARN] coords.json 없음")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
