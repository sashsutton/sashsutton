"""
We're preparing a portrait photo for clean ASCII conversion
"""


import os
import sys
import cv2
import numpy as np
from PIL import Image
from rembg import remove

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

#0 Square-crop around the face. The ascii grid is ~square, so a tall portrait
#  fed in whole gets squashed. Detect the face and frame a head-and-shoulders
#  square around it; fall back to a top-aligned square if detection fails.
src = Image.open(INPUT).convert("RGB")
W, H = src.size
if W != H:
    det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(cv2.cvtColor(np.array(src), cv2.COLOR_RGB2GRAY), 1.1, 6)

    side = min(W, H)
    if len(det):
        fx, fy, fw, fh = max(det, key=lambda f: f[2] * f[3])
        cx = fx + fw / 2.0
        # put the eyeline in the upper third so the shoulders fill the frame
        top = fy - fh * 0.9
    else:
        cx, top = W / 2.0, 0.0

    left = int(round(min(max(cx - side / 2.0, 0), W - side)))
    top = int(round(min(max(top, 0), H - side)))
    src = src.crop((left, top, left + side, top + side))
    print(f"cropped {W}x{H} -> {side}x{side} at ({left},{top})")

#1 Cut out the subject from the background
cut = remove(src.convert("RGBA"))
rgb = np.array(cut.convert("RGB"))
alpha = np.array(cut.split()[-1])                 # 0 = background

#2
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
CLIP = float(os.environ.get("CLIP", 4.5))
TILE = int(os.environ.get("TILE", 10))
clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(TILE, TILE))
gray = clahe.apply(gray)

# Squeeze the subject's tonal range toward the middle of the ascii ramp. A dark
# top on a dark backdrop otherwise spans nearly 0..255, which pins the clothing
# at the dense end (a solid #%@ slab) and blows the face out to blank. CLAHE
# above has already put the local detail in; this only sets where that detail
# lands on the ramp.
TARGET_MEAN = float(os.environ.get("TARGET_MEAN", 140))
SPREAD = float(os.environ.get("SPREAD", 0.80))
subject = alpha > 8
if subject.any():
    g = gray.astype(np.float32)
    g = TARGET_MEAN + (g - g[subject].mean()) * SPREAD
    gray = np.clip(g, 0, 255).astype(np.uint8)

#3 

mask = (alpha.astype(np.float32) / 255.0)
mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
out = np.clip(out, 0, 255).astype(np.uint8)

Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape)





