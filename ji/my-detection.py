import os, time
import numpy as np
import jetson_utils_python as jetson_utils
import jetson_inference
import vlc

# === 오디오 설정 ===
MP3_FILE = "/home/jetson/vigil/ji/lion_roar_3sec.mp3"
AUDIO_LEN_SEC = 3.2             # 3초짜리 파일이면 약간 여유
assert os.path.exists(MP3_FILE), f"MP3 not found: {MP3_FILE}"

# VLC 인스턴스: pulse/alsa 중 환경에 맞게 선택
#  - PulseAudio 쓰면: '--aout=pulse'
#  - ALSA로 강제하려면: '--aout=alsa'
#  - X 없이도 돌도록 '--no-xlib' 권장
vlc_instance = vlc.Instance('--no-xlib', '--no-video', '--aout=pulse')
player = vlc_instance.media_player_new()

def play_mp3_once():
    media = vlc_instance.media_new(MP3_FILE)
    player.set_media(media)
    player.stop()           # 재생 중이었으면 정지
    player.play()
    print("-----------------playing----------------")
    # 짧게 준비 시간(버퍼링) 주면 안정적
    time.sleep(0.05)

# === 라벨 매핑 ===
WIDTH, HEIGHT = 640, 480
LABELS_PATH = "/home/jetson/vigil/ji/labels.txt"
TARGET_NAME = "dog"

def normalize(s):  # 공백/대소문자/언더스코어 정규화
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

# === 카메라/표시/모델 ===
camera = jetson_utils.gstCamera(WIDTH, HEIGHT, "/dev/video0")
# display_local = jetson_utils.videoOutput()
# display_rtmp  = jetson_utils.videoOutput("rtmp://20.249.68.101/live/jetson1")
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

print("=== Starting ===")
print("DISPLAY:", os.environ.get("DISPLAY"))
print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))

# === 타이머/오디오 상태 ===
present_since = None          # 타겟이 보이기 시작한 시각
audio_playing = False         # 현재 음원 재생 중?
audio_started = 0.0           # 재생 시작 시각

start = time.time()
try:
    while True:
        img, w, h = camera.CaptureRGBA()
        if img is None:
            print("Capture failed"); break

        dets = net.Detect(img, w, h)

        # ---- 2초 연속 감지 체크 ----
        now = time.time()
        is_present = any(d.ClassID == target_id for d in dets) if target_id is not None else False

        if is_present:
            if present_since is None:
                present_since = now
            elif (now - present_since) >= 2.0 and not audio_playing:
                # 2초 연속으로 보였고 아직 소리 안나가는 경우 → 재생 시작
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
        # if display_local and display_local.IsStreaming():
        #     display_local.Render(img)
        #     display_local.SetStatus(f"{net.GetNetworkFPS():.0f} FPS | det={len(dets)}")
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
