import os, time
import numpy as np
import jetson_utils_python as jetson_utils
from jetson_inference import poseNet   # ★ detectNet 대신 poseNet 사용
import vlc

# === 오디오 설정 ===
MP3_FILE = "/home/jetson/vigil/ji/lion_roar_3sec.mp3"
AUDIO_LEN_SEC = 3.2             # 3초짜리 파일이면 약간 여유
assert os.path.exists(MP3_FILE), f"MP3 not found: {MP3_FILE}"

# VLC 인스턴스 (환경 맞게 aout 조정)
vlc_instance = vlc.Instance('--no-xlib', '--no-video', '--aout=pulse')
player = vlc_instance.media_player_new()
def play_mp3_once():
    media = vlc_instance.media_new(MP3_FILE)
    player.set_media(media)
    player.stop()
    player.play()
    print("----------------------playing------------------")
    time.sleep(0.05)  # 버퍼링 소량 대기

# === 카메라/표시/모델 ===
WIDTH, HEIGHT = 640, 480
camera = jetson_utils.gstCamera(WIDTH, HEIGHT, "/dev/video0")
display_local = jetson_utils.videoOutput()  # 모니터 직결일 때
display_rtmp  = jetson_utils.videoOutput("rtmp://192.168.68.87/live/jetson1")

# poseNet 로드 (모델/임계값/오버레이)
NETWORK   = "resnet18-body"         # 'resnet18-body' | 'densenet121-body' | 'resnet18-hand'
THRESHOLD = 0.15
OVERLAY   = "links,keypoints"       # 'boxes','links','keypoints' 조합 또는 'none'
net = poseNet(NETWORK, threshold=THRESHOLD)

print("=== Starting ===")
print("DISPLAY:", os.environ.get("DISPLAY"))
print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))

# === 타이머/오디오 상태 ===
present_since = None          # 사람이 보이기 시작한 시각
audio_playing = False
audio_started = 0.0
start = time.time()

try:
    while True:
        img, w, h = camera.CaptureRGBA()
        if img is None:
            print("Capture failed"); break

        # ★ 포즈 추론 + 오버레이
        poses = net.Process(img, overlay=OVERLAY)   # boxes/links/keypoints 중 선택

        # ---- 2초 연속 감지 체크 (사람 1명 이상) ----
        now = time.time()
        is_present = len(poses) > 0

        if is_present:
            if present_since is None:
                present_since = now
            elif (now - present_since) >= 2.0 and not audio_playing:
                play_mp3_once()
                audio_started = time.time()
                audio_playing = True
        else:
            present_since = None  # 끊기면 타이머 리셋

        # ---- 재생 종료 타이밍 ----
        if audio_playing and (time.time() - audio_started) >= AUDIO_LEN_SEC:
            player.stop()
            audio_playing = False

        # ---- 출력 ----
        if display_local and display_local.IsStreaming():
            display_local.Render(img)
            display_local.SetStatus(f"{net.GetNetworkFPS():.0f} FPS | people={len(poses)}")

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
