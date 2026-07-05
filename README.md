# 🤟 Indian Sign Language — Real-Time ML Translation System

> **Real-time Indian Sign Language (ISL) recognition** using a CNN trained on hand-landmark skeletons, with live text output and Text-to-Speech (TTS) feedback.

Built with **Python · TensorFlow · OpenCV · MediaPipe · pyttsx3**

---

## 🎯 What It Does

1. **Detects your hand** live from webcam using MediaPipe landmark detection
2. **Draws the hand skeleton** on a white canvas (making it lighting/background invariant)
3. **Classifies the gesture** using a trained 8-group CNN model achieving **97% accuracy**
4. **Builds words and sentences** from detected characters with spell-check suggestions
5. **Speaks the sentence aloud** via Text-to-Speech (pyttsx3)

---

## ✨ Features

| Feature | Details |
|---------|---------|
| Real-time detection | < 30ms inference per frame |
| Alphabet coverage | A–Z (all 26 ISL fingerspelling gestures) |
| Background invariant | Works in any lighting using skeleton rendering |
| Word suggestions | pyenchant spell-check with 4 live suggestions |
| Text-to-Speech | pyttsx3 TTS, click "Speak" button or auto |
| Gesture commands | Space · Backspace · Next-character gestures |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9.x
- Webcam / built-in camera

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/adarsh-singh07/Indian-Sign-Language-Real-Time-ML-Translation-System.git
cd Indian-Sign-Language-Real-Time-ML-Translation-System

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Main application (Tkinter GUI + TTS)
python final_pred.py

# CLI version (OpenCV window, no GUI)
python prediction_wo_gui.py
```

---

## 🧠 How the Model Works

The system uses a **two-stage pipeline**:

```
Webcam Frame
    ↓
MediaPipe Hand Detection (cvzone wrapper)
    ↓
Crop hand ROI → Draw 21 landmarks on white 400×400 canvas
    ↓
CNN Model (8-group classifier)    ←── cnn8grps_rad1_model.h5
    ↓
Geometric landmark rules → Final letter classification
    ↓
Text buffer → Word suggestions → TTS
```

### Why 8 Groups?
Direct 26-class classification gave poor accuracy on visually similar letters. We grouped similar gestures:

| Group | Letters |
|-------|---------|
| 0 | A, E, M, N, S, T |
| 1 | B, D, F, I, K, R, U, V, W |
| 2 | C, O |
| 3 | G, H |
| 4 | L |
| 5 | P, Q, Z |
| 6 | X |
| 7 | Y, J |

The CNN classifies the group, then geometric rules on hand landmarks resolve the exact letter within the group.

---

## 📁 Project Structure

```
├── final_pred.py              # Main GUI application (run this)
├── prediction_wo_gui.py       # CLI/headless version
├── data_collection_binary.py  # Dataset collection utility
├── data_collection_final.py   # Dataset collection v2
├── cnn8grps_rad1_model.h5     # Trained CNN model weights
├── white.jpg                  # White canvas template for skeleton rendering
├── AtoZ_3.1/                  # Training image dataset (A–Z folders, 180 imgs each)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| CNN model accuracy | **97%** (any background/lighting) |
| Best case (clean BG) | **99%** |
| Training images | 180 skeleton images per letter (A–Z) |
| Input resolution | 400 × 400 px (grayscale skeleton) |

---

## 🛠️ Tech Stack

- **Python 3.9**
- **TensorFlow / Keras** — CNN model inference
- **OpenCV** — video capture, image processing
- **MediaPipe / cvzone** — real-time hand landmark detection
- **pyttsx3** — offline Text-to-Speech
- **pyenchant** — spell-check & word suggestions
- **Tkinter + Pillow** — GUI

---

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| Any ISL letter | Detected and displayed |
| Open-palm sideways | **Space** |
| All fingers extended + thumb tucked | **Next character** (confirm) |
| Thumbs up / fist with thumb up | **Backspace** |
| Click "Speak" button | **TTS reads the sentence** |
| Click "Clear" button | **Clear sentence** |
| Click word suggestion buttons | **Auto-complete word** |

---

## 👨‍💻 Author

**Adarsh Singh** — adarsh2001gop@gmail.com

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
