import time
import os
import vlc

os.environ["PULSE_SINK"] = "alsa_output.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-stereo"

mp3_file = "/home/jetson/vigil/ji/test.mp3"
time.sleep(3)   # 3 seconds
player = vlc.MediaPlayer(mp3_file)
player.play()
print("-------------------------playing---------------")
time.sleep(5)
player.stop()
print("-----------------------stop--------------------")

# 4	alsa_output.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-stereo	module-alsa-card.c	s16le 2ch 44100Hz	SUSPENDED
