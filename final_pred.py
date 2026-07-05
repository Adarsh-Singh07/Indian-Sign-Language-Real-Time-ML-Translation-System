"""
╔══════════════════════════════════════════════════════════════════╗
║   Indian Sign Language — Real-Time ML Translation System        ║
║   Author  : Adarsh Singh  |  github.com/Adarsh-Singh07          ║
║   UI      : CustomTkinter — dark glassmorphic Apple-style theme  ║
║   Inference: Threaded CNN worker (non-blocking, smooth 30 fps)   ║
║   TTS     : Background pyttsx3 worker (never blocks UI)          ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Suppress TensorFlow, Keras, and MediaPipe log noise before imports
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ─── Standard library ─────────────────────────────────────────────────────────
import math
import threading
import queue
import traceback
import tkinter as tk
from string import ascii_uppercase

# ─── Third-party ──────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pyttsx3
import enchant
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import customtkinter as ctk
from PIL import Image, ImageTk

# ─── Paths (relative to this file — works on any machine) ─────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "cnn8grps_rad1_model.h5")
WHITE_PATH  = os.path.join(BASE_DIR, "white.jpg")
BOX_OFFSET  = 29

# ─── CustomTkinter global theme ───────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Glassmorphic colour palette ──────────────────────────────────────────────
BG            = "#070714"    # deepest background
GLASS_1       = "#0d0d20"    # card surface
GLASS_2       = "#12122a"    # slightly elevated surface
BORDER        = "#1f1f45"    # subtle border / glow
BORDER_BRIGHT = "#3a3a80"    # highlighted border
ACCENT        = "#7b6fff"    # primary purple
ACCENT_DARK   = "#5548cc"    # pressed / hover purple
TEAL          = "#00d4aa"    # secondary green-teal
TEAL_DARK     = "#009977"    # teal hover
DANGER        = "#ff4466"    # clear / warning
DANGER_BG     = "#1f0a12"    # danger button bg
TEXT          = "#eeeeff"    # primary text
TEXT_DIM      = "#6666aa"    # muted labels
TEXT_BRIGHT   = "#ffffff"    # pure white (char display)
FLASH_GREEN   = "#00ff99"    # new-char flash colour


# ══════════════════════════════════════════════════════════════════════════════
# TTS WORKER — speaks text in a background thread so UI never freezes
# ══════════════════════════════════════════════════════════════════════════════
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._q      = queue.Queue()
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 145)
        voices = self._engine.getProperty("voices")
        if voices:
            self._engine.setProperty("voice", voices[0].id)

    def say(self, text: str):
        """Queue text for speaking (non-blocking)."""
        if text.strip():
            self._q.put(text.strip())

    def run(self):
        while True:
            text = self._q.get()
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass   # keep worker alive even on TTS errors


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE WORKER — runs CNN model.predict() off the main thread
# ══════════════════════════════════════════════════════════════════════════════
class InferenceWorker(threading.Thread):
    """
    Uses size-1 queues so stale frames are silently dropped and we always
    process the most recent skeleton image.  Keeps the UI at a smooth 30 fps.
    """

    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self._in   = queue.Queue(maxsize=1)
        self._out  = queue.Queue(maxsize=1)

    def submit(self, white_img: np.ndarray, pts: list):
        """Non-blocking — drop old frame when worker is busy."""
        try:
            self._in.put_nowait((white_img.copy(), list(pts)))
        except queue.Full:
            pass

    def get_result(self):
        """Returns (ch1, confidence_pct) or None if no result ready."""
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def run(self):
        while True:
            white_img, pts = self._in.get()
            try:
                result = self._infer(white_img, pts)
                try:
                    self._out.put_nowait(result)
                except queue.Full:
                    pass
            except Exception:
                pass   # protect worker thread from crashing

    # ── Euclidean distance helper ──────────────────────────────────────────────
    @staticmethod
    def _d(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # ── Full inference: CNN + geometric landmark disambiguation rules ───────────
    def _infer(self, white, pts):
        img_r = white.reshape(1, 400, 400, 3)
        prob  = np.array(self.model.predict(img_r, verbose=0)[0], dtype="float32")
        conf  = float(prob.max())                    # top-class raw confidence
        ch1   = int(np.argmax(prob)); prob[ch1] = 0
        ch2   = int(np.argmax(prob)); prob[ch2] = 0

        d  = self._d
        pl = [ch1, ch2]

        # ── 8-group → sub-group disambiguation rules ───────────────────────────
        # [Aemnst]
        l = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],
             [6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],
             [1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],
             [5,4],[7,0],[7,5],[7,2]]
        if pl in l:
            if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]:
                ch1=0

        l = [[2,2],[2,1]]
        if pl in l:
            if pts[5][0]<pts[4][0]: ch1=0

        l = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
        pl=[ch1,ch2]
        if pl in l:
            if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and
                    pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0]>pts[4][0]:
                ch1=2

        l=[[6,0],[6,6],[6,2]]; pl=[ch1,ch2]
        if pl in l:
            if d(pts[8],pts[16])<52: ch1=2

        l=[[1,4],[1,5],[1,6],[1,3],[1,0]]; pl=[ch1,ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1] and
                    pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]):
                ch1=3

        l=[[4,6],[4,1],[4,5],[4,3],[4,7]]; pl=[ch1,ch2]
        if pl in l:
            if pts[4][0]>pts[0][0]: ch1=3

        l=[[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]]; pl=[ch1,ch2]
        if pl in l:
            if pts[2][1]+15<pts[16][1]: ch1=3

        l=[[6,4],[6,1],[6,2]]; pl=[ch1,ch2]
        if pl in l:
            if d(pts[4],pts[11])>55: ch1=4

        l=[[1,4],[1,6],[1,1]]; pl=[ch1,ch2]
        if pl in l:
            if d(pts[4],pts[11])>50 and (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
                ch1=4

        l=[[3,6],[3,4]]; pl=[ch1,ch2]
        if pl in l:
            if pts[4][0]<pts[0][0]: ch1=4

        l=[[2,2],[2,5],[2,4]]; pl=[ch1,ch2]
        if pl in l:
            if pts[1][0]<pts[12][0]: ch1=4

        l=[[3,6],[3,5],[3,4]]; pl=[ch1,ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[10][1]:
                ch1=5

        l=[[3,2],[3,1],[3,6]]; pl=[ch1,ch2]
        if pl in l:
            if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]:
                ch1=5

        l=[[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]; pl=[ch1,ch2]
        if pl in l:
            if pts[4][0]>pts[0][0]: ch1=5

        l=[[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]; pl=[ch1,ch2]
        if pl in l:
            if pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
                ch1=5

        l=[[5,7],[5,2],[5,6]]; pl=[ch1,ch2]
        if pl in l:
            if pts[3][0]<pts[0][0]: ch1=7

        l=[[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]]; pl=[ch1,ch2]
        if pl in l:
            if pts[6][1]<pts[8][1]: ch1=7

        l=[[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]; pl=[ch1,ch2]
        if pl in l:
            if pts[18][1]>pts[20][1]: ch1=7

        l=[[0,4],[0,2],[0,3],[0,1],[0,6]]; pl=[ch1,ch2]
        if pl in l:
            if pts[5][0]>pts[16][0]: ch1=6

        l=[[7,2]]; pl=[ch1,ch2]
        if pl in l:
            if pts[18][1]<pts[20][1] and pts[8][1]<pts[10][1]: ch1=6

        l=[[2,1],[2,2],[2,6],[2,7],[2,0]]; pl=[ch1,ch2]
        if pl in l:
            if d(pts[8],pts[16])>50: ch1=6

        l=[[4,6],[4,2],[4,1],[4,4]]; pl=[ch1,ch2]
        if pl in l:
            if d(pts[4],pts[11])<60: ch1=6

        l=[[1,4],[1,6],[1,0],[1,2]]; pl=[ch1,ch2]
        if pl in l:
            if pts[5][0]-pts[4][0]-15>0: ch1=6

        l=[[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]; pl=[ch1,ch2]
        if pl in l:
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                ch1=1

        l=[[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],[4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5],[7,2]]; pl=[ch1,ch2]
        if pl in l:
            if pts[6][1]<pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                ch1=1

        l=[[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]]; pl=[ch1,ch2]
        if pl in l:
            if pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                ch1=1

        l=[[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]; pl=[ch1,ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[2][0]<pts[0][0] and pts[4][1]>pts[14][1]:
                ch1=1

        l=[[4,1],[4,2],[4,4]]; pl=[ch1,ch2]
        if pl in l:
            if d(pts[4],pts[11])<50 and (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
                ch1=1

        l=[[3,4],[3,0],[3,1],[3,5],[3,6]]; pl=[ch1,ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[2][0]<pts[0][0] and pts[14][1]<pts[4][1]:
                ch1=1

        l=[[6,6],[6,4],[6,1],[6,2]]; pl=[ch1,ch2]
        if pl in l:
            if pts[5][0]-pts[4][0]-15<0: ch1=1

        l=[[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],[7,6],[7,7]]; pl=[ch1,ch2]
        if pl in l:
            if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]:
                ch1=1

        l=[[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]; pl=[ch1,ch2]
        if pl in l:
            if pts[4][0]<pts[5][0]+15 and (pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]):
                ch1=7

        l=[[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]; pl=[ch1,ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[14][1]:
                ch1=1

        l=[[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]; pl=[ch1,ch2]
        if pl in l:
            fg=13
            if (not(pts[0][0]+fg<pts[8][0] and pts[0][0]+fg<pts[12][0] and pts[0][0]+fg<pts[16][0] and pts[0][0]+fg<pts[20][0]) and
                not(pts[0][0]>pts[8][0]  and pts[0][0]>pts[12][0]  and pts[0][0]>pts[16][0]  and pts[0][0]>pts[20][0]) and
                d(pts[4],pts[11])<50):
                ch1=1

        l=[[5,0],[5,5],[0,1]]; pl=[ch1,ch2]
        if pl in l:
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1]:
                ch1=1

        # ── Sub-group → single letter ──────────────────────────────────────────
        if ch1==0:
            ch1='S'
            if pts[4][0]<pts[6][0] and pts[4][0]<pts[10][0] and pts[4][0]<pts[14][0] and pts[4][0]<pts[18][0]: ch1='A'
            if pts[4][0]>pts[6][0] and pts[4][0]<pts[10][0] and pts[4][0]<pts[14][0] and pts[4][0]<pts[18][0] and pts[4][1]<pts[14][1] and pts[4][1]<pts[18][1]: ch1='T'
            if pts[4][1]>pts[8][1] and pts[4][1]>pts[12][1] and pts[4][1]>pts[16][1] and pts[4][1]>pts[20][1]: ch1='E'
            if pts[4][0]>pts[6][0] and pts[4][0]>pts[10][0] and pts[4][0]>pts[14][0] and pts[4][1]<pts[18][1]: ch1='M'
            if pts[4][0]>pts[6][0] and pts[4][0]>pts[10][0] and pts[4][1]<pts[18][1] and pts[4][1]<pts[14][1]: ch1='N'

        if ch1==2: ch1='C' if d(pts[12],pts[4])>42 else 'O'
        if ch1==3: ch1='G' if d(pts[8],pts[12])>72 else 'H'
        if ch1==7: ch1='Y' if d(pts[8],pts[4])>42 else 'J'
        if ch1==4: ch1='L'
        if ch1==6: ch1='X'
        if ch1==5:
            if pts[4][0]>pts[12][0] and pts[4][0]>pts[16][0] and pts[4][0]>pts[20][0]:
                ch1='Z' if pts[8][1]<pts[5][1] else 'Q'
            else:
                ch1='P'

        if ch1==1:
            if pts[6][1]>pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]: ch1='B'
            if pts[6][1]>pts[8][1]  and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]: ch1='D'
            if pts[6][1]<pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]: ch1='F'
            if pts[6][1]<pts[8][1]  and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]: ch1='I'
            if pts[6][1]>pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]<pts[20][1]: ch1='W'
            if (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]<pts[9][1]: ch1='K'
            if (d(pts[8],pts[12])-d(pts[6],pts[10]))<8  and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]): ch1='U'
            if (d(pts[8],pts[12])-d(pts[6],pts[10]))>=8 and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[9][1]: ch1='V'
            if pts[8][0]>pts[12][0] and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]): ch1='R'

        # ── Special gestures ───────────────────────────────────────────────────
        # Space (index+pinky up, middle+ring down)
        if ch1 in [1, 'E', 'S', 'X', 'Y', 'B']:
            if pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]:
                ch1=' '

        # Next (confirm character — all fingers up, thumb tucked)
        if ch1 in ['E', 'Y', 'B']:
            if (pts[4][0]<pts[5][0] and pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and
                    pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]):
                ch1='next'

        # Backspace (thumb-up fist)
        if ch1 in ['Next', 'B', 'C', 'H', 'F', 'X']:
            if (pts[0][0]>pts[8][0]  and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0] and
                    pts[4][1]<pts[8][1]  and pts[4][1]<pts[12][1] and pts[4][1]<pts[16][1] and pts[4][1]<pts[20][1] and
                    pts[4][1]<pts[6][1]  and pts[4][1]<pts[10][1] and pts[4][1]<pts[14][1] and pts[4][1]<pts[18][1]):
                ch1='Backspace'

        return ch1, round(conf * 100, 1)


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class Application:

    def __init__(self):
        # ── Load model & assets ────────────────────────────────────────────────
        print("Loading CNN model…")
        self.model = load_model(MODEL_PATH)

        self.white_template = cv2.imread(WHITE_PATH)
        if self.white_template is None:
            raise FileNotFoundError(f"white.jpg not found at: {WHITE_PATH}")

        # ── Hand detectors ─────────────────────────────────────────────────────
        self.hd  = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)

        # ── Spell-check ────────────────────────────────────────────────────────
        self.ddd = enchant.Dict("en-US")

        # ── Background workers ─────────────────────────────────────────────────
        self.tts = TTSWorker();   self.tts.start()
        self.inf = InferenceWorker(self.model); self.inf.start()

        # ── Camera ─────────────────────────────────────────────────────────────
        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            raise RuntimeError("Cannot open webcam — please connect a camera and retry.")

        # ── Character buffer state ─────────────────────────────────────────────
        self.sentence   = ""
        self.word       = ""
        self.word1 = self.word2 = self.word3 = self.word4 = ""
        self.prev_char  = ""
        self.count      = -1
        self.ccc        = 0
        self.ten_prev   = [" "] * 10
        self.pts        = []

        # ── Build UI, then schedule first frame ────────────────────────────────
        self._build_ui()
        self.root.after(100, self._video_loop)

    # ══════════════════════════════════════════════════════════════════════════
    # UI LAYOUT  (glassmorphic dark — Apple-style)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.root = ctk.CTk()
        self.root.title("ISL Sign Language → Text & Speech")
        self.root.geometry("1380x820")
        self.root.configure(fg_color=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._destructor)

        # ── Header ─────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self.root, fg_color=GLASS_1, corner_radius=0,
                           border_width=1, border_color=BORDER, height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Left: logo + title
        hdr_l = ctk.CTkFrame(hdr, fg_color="transparent")
        hdr_l.pack(side="left", padx=20, pady=8)
        ctk.CTkLabel(hdr_l, text="🤟",
                     font=ctk.CTkFont("Segoe UI", 28)).pack(side="left", padx=(0,10))
        ctk.CTkLabel(hdr_l, text="Indian Sign Language  —  Real-Time ML Translator",
                     font=ctk.CTkFont("Segoe UI", 18, "bold"),
                     text_color=TEXT).pack(side="left")

        # Right: live indicator dot
        hdr_r = ctk.CTkFrame(hdr, fg_color="transparent")
        hdr_r.pack(side="right", padx=24)
        ctk.CTkLabel(hdr_r, text="● LIVE",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=TEAL).pack()

        # ── Body ───────────────────────────────────────────────────────────────
        body = ctk.CTkFrame(self.root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=12)

        # ── LEFT COLUMN: Camera + Skeleton ─────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=GLASS_1, corner_radius=20,
                            border_width=1, border_color=BORDER, width=550)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        # Camera label
        ctk.CTkLabel(left, text="📷  Live Camera Feed",
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=TEXT_DIM).pack(pady=(14, 4))

        # Camera canvas
        cam_frame = ctk.CTkFrame(left, fg_color=GLASS_2, corner_radius=14,
                                 border_width=1, border_color=BORDER)
        cam_frame.pack(padx=16, pady=(0, 10))
        self.cam_canvas = tk.Canvas(cam_frame, width=510, height=384,
                                    bg="#050510", highlightthickness=0)
        self.cam_canvas.pack(padx=4, pady=4)

        # Skeleton label + canvas
        ctk.CTkLabel(left, text="🦴  Hand Skeleton (CNN Input)",
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=TEXT_DIM).pack()

        skel_frame = ctk.CTkFrame(left, fg_color=GLASS_2, corner_radius=14,
                                  border_width=1, border_color=BORDER)
        skel_frame.pack(padx=16, pady=(4, 14))
        self.skel_canvas = tk.Canvas(skel_frame, width=230, height=230,
                                     bg="#050510", highlightthickness=0)
        self.skel_canvas.pack(padx=4, pady=4)

        # ── RIGHT COLUMN ────────────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        # ── Card 1: Detected Character ─────────────────────────────────────────
        card_char = ctk.CTkFrame(right, fg_color=GLASS_1, corner_radius=20,
                                 border_width=1, border_color=BORDER)
        card_char.pack(fill="x", pady=(0, 10))

        inner = ctk.CTkFrame(card_char, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=16)

        ctk.CTkLabel(inner, text="DETECTED CHARACTER",
                     font=ctk.CTkFont("Segoe UI", 9, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w")

        row = ctk.CTkFrame(inner, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))

        # Big character glyph
        self.char_label = ctk.CTkLabel(row, text="—",
                                       font=ctk.CTkFont("Segoe UI", 80, "bold"),
                                       text_color=ACCENT, width=110, anchor="center")
        self.char_label.pack(side="left")

        # Confidence info
        conf_block = ctk.CTkFrame(row, fg_color="transparent")
        conf_block.pack(side="left", padx=28, fill="y")

        ctk.CTkLabel(conf_block, text="Confidence",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=TEXT_DIM).pack(anchor="w")
        self.conf_pct = ctk.CTkLabel(conf_block, text="—",
                                     font=ctk.CTkFont("Segoe UI", 34, "bold"),
                                     text_color=TEAL)
        self.conf_pct.pack(anchor="w")
        self.conf_bar = ctk.CTkProgressBar(conf_block, width=220, height=8,
                                           fg_color=BORDER, progress_color=TEAL,
                                           corner_radius=4)
        self.conf_bar.set(0)
        self.conf_bar.pack(anchor="w", pady=(6, 0))

        # Gesture hint
        ctk.CTkLabel(conf_block, text="✋  Space  |  👍  Next  |  🤜  Backspace",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=TEXT_DIM).pack(anchor="w", pady=(10, 0))

        # ── Card 2: Sentence ───────────────────────────────────────────────────
        card_sent = ctk.CTkFrame(right, fg_color=GLASS_1, corner_radius=20,
                                 border_width=1, border_color=BORDER)
        card_sent.pack(fill="x", pady=(0, 10))

        inner2 = ctk.CTkFrame(card_sent, fg_color="transparent")
        inner2.pack(fill="x", padx=22, pady=16)

        ctk.CTkLabel(inner2, text="SENTENCE",
                     font=ctk.CTkFont("Segoe UI", 9, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w")

        self.sent_label = ctk.CTkLabel(inner2, text="Start signing to build a sentence…",
                                       font=ctk.CTkFont("Segoe UI", 24, "bold"),
                                       text_color=TEXT_DIM, wraplength=580,
                                       justify="left", anchor="w")
        self.sent_label.pack(fill="x", pady=(8, 0))

        # ── Card 3: Word suggestions ───────────────────────────────────────────
        card_sugg = ctk.CTkFrame(right, fg_color=GLASS_1, corner_radius=20,
                                 border_width=1, border_color=BORDER)
        card_sugg.pack(fill="x", pady=(0, 10))

        inner3 = ctk.CTkFrame(card_sugg, fg_color="transparent")
        inner3.pack(fill="x", padx=22, pady=14)

        ctk.CTkLabel(inner3, text="WORD SUGGESTIONS",
                     font=ctk.CTkFont("Segoe UI", 9, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", pady=(0, 8))

        sugg_row = ctk.CTkFrame(inner3, fg_color="transparent")
        sugg_row.pack(fill="x")

        self._sugg_btns = []
        for fn in [self._action1, self._action2, self._action3, self._action4]:
            b = ctk.CTkButton(sugg_row, text="…",
                              font=ctk.CTkFont("Segoe UI", 13, "bold"),
                              fg_color=GLASS_2, hover_color=ACCENT_DARK,
                              text_color=TEXT, border_width=1, border_color=BORDER,
                              corner_radius=12, height=40, command=fn)
            b.pack(side="left", expand=True, fill="x", padx=(0, 6))
            self._sugg_btns.append(b)

        # ── Card 4: Controls ───────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(right, fg_color="transparent")
        ctrl.pack(fill="x", pady=(0, 4))

        self.speak_btn = ctk.CTkButton(
            ctrl, text="🔊   Speak Sentence",
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_DARK,
            corner_radius=14, height=54, command=self._speak)
        self.speak_btn.pack(side="left", expand=True, fill="x", padx=(0, 8))

        self.clear_btn = ctk.CTkButton(
            ctrl, text="🗑️   Clear",
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
            fg_color=DANGER_BG, hover_color=DANGER,
            text_color=DANGER, border_width=1, border_color=DANGER,
            corner_radius=14, height=54, command=self._clear)
        self.clear_btn.pack(side="left", expand=True, fill="x")

        # ── Status bar ─────────────────────────────────────────────────────────
        status = ctk.CTkFrame(self.root, fg_color=GLASS_1, corner_radius=0,
                              border_width=1, border_color=BORDER, height=28)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        self._status_lbl = ctk.CTkLabel(status, text="Ready — show your hand to the camera",
                                        font=ctk.CTkFont("Segoe UI", 10),
                                        text_color=TEXT_DIM)
        self._status_lbl.pack(side="left", padx=14, pady=4)
        ctk.CTkLabel(status, text="ISL ML Translator  v2.0",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=TEXT_DIM).pack(side="right", padx=14)

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO LOOP  (runs every 1 ms via root.after — camera reads ~30 fps)
    # ══════════════════════════════════════════════════════════════════════════
    def _video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                self.root.after(10, self._video_loop)
                return

            frame = cv2.flip(frame, 1)

            # ── Hand detection (main thread — fast, <10 ms) ────────────────────
            hands     = self.hd.findHands(frame, draw=False, flipType=True)
            hand_list = hands[0] if isinstance(hands, tuple) else hands

            if hand_list:
                hand = hand_list[0]
                x, y, w, h = hand["bbox"]

                # Clip ROI to frame boundaries
                y1 = max(0, y-BOX_OFFSET);  y2 = min(frame.shape[0], y+h+BOX_OFFSET)
                x1 = max(0, x-BOX_OFFSET);  x2 = min(frame.shape[1], x+w+BOX_OFFSET)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    white = self.white_template.copy()
                    handz     = self.hd2.findHands(crop, draw=False, flipType=True)
                    hand_list2= handz[0] if isinstance(handz, tuple) else handz
                    if hand_list2:
                        self.pts = hand_list2[0]["lmList"]
                        ox = ((400-w)//2)-15
                        oy = ((400-h)//2)-15
                        self._draw_skeleton(white, ox, oy)
                        # Send to inference thread (non-blocking drop if busy)
                        self.inf.submit(white, self.pts)
                        self._render_skeleton(white)

                # Draw bounding box on live frame
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (123, 111, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Hand Detected", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123,111,255), 1, cv2.LINE_AA)
                self._status("🤚  Hand detected — inferring gesture…")
            else:
                self._status("👁️  Waiting for hand — hold ISL sign to camera")

            # ── Pull latest inference result (non-blocking) ────────────────────
            result = self.inf.get_result()
            if result is not None:
                ch1, conf = result
                self._update_buffer(ch1)
                self._update_char_ui(ch1, conf)

            # ── Display camera feed ────────────────────────────────────────────
            self._render_camera(frame)

        except Exception:
            print("[video_loop error]\n", traceback.format_exc())
        finally:
            self.root.after(1, self._video_loop)

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_skeleton(self, white, ox, oy):
        """Draw 21-point hand skeleton onto white canvas (CPU-only, fast)."""
        p = self.pts
        segs = [(0,4),(5,8),(9,12),(13,16),(17,20)]
        for s,e in segs:
            for t in range(s, e):
                cv2.line(white,(p[t][0]+ox,p[t][1]+oy),(p[t+1][0]+ox,p[t+1][1]+oy),(0,255,0),3)
        # Palm
        for a,b in [(5,9),(9,13),(13,17),(0,5),(0,17)]:
            cv2.line(white,(p[a][0]+ox,p[a][1]+oy),(p[b][0]+ox,p[b][1]+oy),(0,255,0),3)
        # Landmark dots - Must match training dataset format exactly:
        # BGR color (0, 0, 255) (Red), radius 2, thickness 1
        for i in range(21):
            cv2.circle(white, (p[i][0]+ox, p[i][1]+oy), 2, (0, 0, 255), 1)

    def _render_camera(self, frame):
        """Resize BGR frame → display on camera canvas."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb).resize((510, 384), Image.BILINEAR)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_canvas.imgtk = imgtk
        self.cam_canvas.create_image(0, 0, anchor="nw", image=imgtk)

    def _render_skeleton(self, white):
        """Resize skeleton image → display on skeleton canvas."""
        rgb   = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb).resize((230, 230), Image.BILINEAR)
        imgtk = ImageTk.PhotoImage(image=img)
        self.skel_canvas.imgtk = imgtk
        self.skel_canvas.create_image(0, 0, anchor="nw", image=imgtk)

    # ──────────────────────────────────────────────────────────────────────────
    def _update_char_ui(self, ch1, conf):
        """Update character glyph, confidence bar, sentence, suggestions."""
        # Character display with colour coding
        char_str = str(ch1).upper() if ch1 not in (None, "", 1) else "—"
        if char_str == " ": char_str = "SPC"
        col = TEAL if char_str not in ["—","NEXT","BACKSPACE","SPC"] else TEXT_DIM
        self.char_label.configure(text=char_str, text_color=col)

        # Confidence
        self.conf_pct.configure(text=f"{conf}%")
        self.conf_bar.set(min(conf / 100.0, 1.0))

        # Sentence (flash accent colour momentarily when new char arrives)
        disp = self.sentence if self.sentence.strip() else ""
        self.sent_label.configure(
            text=disp if disp else "Start signing to build a sentence…",
            text_color=TEXT if disp else TEXT_DIM)

        # Suggestions
        words = [self.word1, self.word2, self.word3, self.word4]
        for btn, w in zip(self._sugg_btns, words):
            btn.configure(text=w.strip().title() if w.strip() else "…")

    # ══════════════════════════════════════════════════════════════════════════
    # CHARACTER BUFFER LOGIC
    # ══════════════════════════════════════════════════════════════════════════
    def _update_buffer(self, ch1):
        """Append characters, handle space/backspace/next, update suggestions."""

        # ── "next" gesture → commit the buffered character ─────────────────────
        if ch1 == "next" and self.prev_char != "next":
            prev_idx = (self.count - 2) % 10
            c = self.ten_prev[prev_idx]
            if isinstance(c, str):
                if c != "next":
                    if c == "Backspace":
                        self.sentence = self.sentence[:-1]
                    elif len(c) == 1:
                        self.sentence += c
            else:
                # If c was not a string (e.g. group index int), check the current character instead
                c2 = self.ten_prev[self.count % 10]
                if isinstance(c2, str) and c2 != "Backspace" and len(c2) == 1:
                    self.sentence += c2

        # ── Space gesture → add space + auto-speak completed word ──────────────
        if ch1 == " " and self.prev_char != " ":
            self.sentence += " "
            words = self.sentence.strip().split()
            if words:
                self.tts.say(words[-1])   # auto-speak the word just finished

        self.prev_char = ch1
        self.count    += 1
        self.ten_prev[self.count % 10] = ch1

        # ── Spell-check suggestions ────────────────────────────────────────────
        if self.sentence.strip():
            idx  = self.sentence.rfind(" ")
            cur  = self.sentence[idx+1:].strip()
            self.word = cur
            if cur:
                sugg = self.ddd.suggest(cur)
                n    = len(sugg)
                self.word1 = sugg[0] if n>=1 else ""
                self.word2 = sugg[1] if n>=2 else ""
                self.word3 = sugg[2] if n>=3 else ""
                self.word4 = sugg[3] if n>=4 else ""
            else:
                self.word1=self.word2=self.word3=self.word4=""

    # ══════════════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    def _replace_last_word(self, new_word):
        if not new_word.strip():
            return
        idx = self.sentence.rfind(" ")
        self.sentence = self.sentence[:idx+1] + new_word.upper()

    def _action1(self): self._replace_last_word(self.word1)
    def _action2(self): self._replace_last_word(self.word2)
    def _action3(self): self._replace_last_word(self.word3)
    def _action4(self): self._replace_last_word(self.word4)

    def _speak(self):
        self.tts.say(self.sentence.strip() or "Nothing to speak yet")

    def _clear(self):
        self.sentence  = ""
        self.word      = ""
        self.word1=self.word2=self.word3=self.word4=""
        self.sent_label.configure(text="Start signing to build a sentence…", text_color=TEXT_DIM)

    def _status(self, msg: str):
        self._status_lbl.configure(text=msg)

    def _destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  Indian Sign Language — Real-Time ML Translation System")
    print("  github.com/Adarsh-Singh07/Indian-Sign-Language-Real-Time-ML-Translation-System")
    print("=" * 62)
    app = Application()
    app.run()
