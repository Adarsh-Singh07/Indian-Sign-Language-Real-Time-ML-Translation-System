# 🤟 Indian Sign Language — Real-Time Web Translation System

> **Real-time Indian Sign Language (ISL) recognition** running entirely in the browser using a CNN model and MediaPipe hand tracking, with live text output and Text-to-Speech (TTS) feedback.

This is the Web Version of the project. The original Python desktop codebase is archived on the `python` branch of this repository.

---

## 🎯 What It Does

1. **Detects your hand** live from webcam in the browser using MediaPipe and WebAssembly.
2. **Draws the hand skeleton** on a white 400x400 canvas (making it lighting/background invariant).
3. **Classifies the gesture** using the converted TensorFlow.js 8-group CNN model.
4. **Applies geometric rules** directly in JavaScript to resolve visually similar letters within groups.
5. **Builds words and sentences** from detected characters with dynamic autocomplete/spell-check suggestions.
6. **Speaks the sentence aloud** via the browser's native Web Speech API (speechSynthesis).

---

## ✨ Web Features

*   **Zero Lag & High Frame Rate**: Tracking runs client-side in WebAssembly using GPU acceleration.
*   **Fully Offline**: Runs directly in the browser without sending your webcam feed to any external server.
*   **Dynamic Spell-Check**: Uses the Datamuse API to suggest words based on the character buffer.
*   **Text-to-Speech**: Browser-native TTS sounds natural and works without third-party API keys.
*   **Responsive Apple-Style Dark Theme**: Sleek glassmorphic layout that scales beautifully.

---

## 🚀 How to Run Locally

Since the app loads the TensorFlow.js model dynamically using `fetch()`, it must be served from a local web server (to bypass browser CORS policies):

1. Clone the repository and switch to the `main` branch.
2. Run a local server from the root directory:
   * **Python**: `python -m http.server 8000`
   * **NodeJS**: `npx serve` or `npx live-server`
3. Open `http://localhost:8000` in your web browser.

---

## 🧠 Model & Architecture

The system uses a **two-stage pipeline**:
1. **CNN Group Predictor**: Classifies the hand skeleton into one of 8 gesture groups.
2. **Geometric Heuristics**: Evaluates exact joint angles and distances to determine the target letter within that group.

### The 8 Gesture Groups
*   **Group 0**: A, E, M, N, S, T
*   **Group 1**: B, D, F, I, K, R, U, V, W
*   **Group 2**: C, O
*   **Group 3**: G, H
*   **Group 4**: L
*   **Group 5**: P, Q, Z
*   **Group 6**: X
*   **Group 7**: J, Y
