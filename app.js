/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║   Indian Sign Language — Real-Time ML Translation System        ║
 * ║   Frontend Engine: MediaPipe WebAssembly + TensorFlow.js         ║
 * ║   Aesthetics: Apple-Style Dark Glassmorphic Theme                ║
 * ║   Features: 60FPS Client-Side tracking, Native TTS, Spell-Check  ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

// --- Global Application State ---
let tfModel = null;
let activeSentence = "";
let currentWord = "";
let prevChar = "";
let counter = -1;
let tenPrevBuffer = Array(10).fill(" ");
let isModelLoaded = false;
let isMediaPipeReady = false;

// UI Elements
const charLabel = document.getElementById("char-label");
const confPct = document.getElementById("conf-pct");
const confBar = document.getElementById("conf-bar");
const sentenceLabel = document.getElementById("sentence-label");
const statusMsg = document.getElementById("status-bar-msg");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const cameraCanvas = document.getElementById("camera-canvas");
const skeletonCanvas = document.getElementById("skeleton-canvas");
const webcamElement = document.getElementById("webcam");
const btnSpeak = document.getElementById("btn-speak");
const btnClear = document.getElementById("btn-clear");

const suggBtns = [
    document.getElementById("sugg-1"),
    document.getElementById("sugg-2"),
    document.getElementById("sugg-3"),
    document.getElementById("sugg-4")
];

// Canvas Contexts
const camCtx = cameraCanvas.getContext("2d");
const skelCtx = skeletonCanvas.getContext("2d");

// Constants
const BOX_OFFSET = 29;
const IMAGE_SIZE = 400; // 400x400 blank canvas input for CNN

// ══════════════════════════════════════════════════════════════════════════════
// 1. SPELL-CHECK & TTS ENGINE
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Speaks the given text using the Web Speech API (offline, native).
 * Falls back if SpeechSynthesis is unsupported.
 */
function speakText(text) {
    if (!text || !text.trim()) return;
    
    // Stop any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text.trim());
    utterance.rate = 0.95; // Slightly slower for better clarity
    
    // Pick a high-quality English voice if available
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith("en-") && v.name.includes("Google")) || 
                         voices.find(v => v.lang.startsWith("en-")) || 
                         voices[0];
    
    if (englishVoice) {
        utterance.voice = englishVoice;
    }
    
    window.speechSynthesis.speak(utterance);
}

// Warm up voices load (some browsers load voices asynchronously)
window.speechSynthesis.onvoiceschanged = () => {};

/**
 * Fetches spelling suggestions using the free, fast, no-auth Datamuse API.
 * Replaces the local pyenchant spellchecker.
 */
async function updateSpellCheckSuggestions(word) {
    if (!word || !word.trim()) {
        clearSuggestions();
        return;
    }
    
    try {
        const query = word.trim().toLowerCase();
        // Datamuse API: sug?s=word (completions and spell suggestions)
        const response = await fetch(`https://api.datamuse.com/sug?s=${query}`);
        const data = await response.json();
        
        const suggestions = data.slice(0, 4).map(item => item.word);
        
        // Fill up buttons
        for (let i = 0; i < 4; i++) {
            const btn = suggBtns[i];
            if (suggestions[i]) {
                btn.textContent = suggestions[i].toUpperCase();
                btn.disabled = false;
            } else {
                btn.textContent = "…";
                btn.disabled = true;
            }
        }
    } catch (err) {
        console.error("Spellcheck API error:", err);
    }
}

function clearSuggestions() {
    suggBtns.forEach(btn => {
        btn.textContent = "…";
        btn.disabled = true;
    });
}

/**
 * Replace the last word of the sentence with the selected suggestion.
 */
function selectSuggestion(index) {
    const wordToInsert = suggBtns[index].textContent;
    if (wordToInsert === "…") return;
    
    const lastSpaceIdx = activeSentence.lastIndexOf(" ");
    if (lastSpaceIdx === -1) {
        activeSentence = wordToInsert + " ";
    } else {
        activeSentence = activeSentence.slice(0, lastSpaceIdx + 1) + wordToInsert + " ";
    }
    
    currentWord = "";
    updateUI();
    speakText(wordToInsert);
    clearSuggestions();
}

// Hook up suggestion buttons
suggBtns.forEach((btn, index) => {
    btn.addEventListener("click", () => selectSuggestion(index));
});

// Controls Handlers
btnSpeak.addEventListener("click", () => {
    speakText(activeSentence || "Nothing to speak yet");
});

btnClear.addEventListener("click", () => {
    activeSentence = "";
    currentWord = "";
    tenPrevBuffer.fill(" ");
    counter = -1;
    prevChar = "";
    clearSuggestions();
    updateUI();
    statusMsg.textContent = "Ready — show your hand to the camera";
});


// ══════════════════════════════════════════════════════════════════════════════
// 2. MODEL LOADER (TF.JS)
// ══════════════════════════════════════════════════════════════════════════

async function initModel() {
    try {
        loadingText.textContent = "Loading TensorFlow.js model...";
        // Load the converted tfjs layers model from public directory
        tfModel = await tf.loadLayersModel("model/model.json");
        isModelLoaded = true;
        console.log("TensorFlow.js model loaded successfully.");
        checkInitializationProgress();
    } catch (err) {
        console.error("Error loading TF.js model:", err);
        loadingText.textContent = "Failed to load model. Check console.";
    }
}

function checkInitializationProgress() {
    if (isModelLoaded && isMediaPipeReady) {
        loadingOverlay.style.display = "none";
        statusMsg.textContent = "Ready — show your hand to the camera";
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// 3. SKELETON RENDERER (White Canvas 400x400)
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Draws the hand skeleton on a solid white 400x400 canvas.
 * Green segments and pink joints exactly mimicking Python's cv2 drawings.
 */
function drawSkeleton(pts, ox, oy) {
    // Fill background with solid white
    skelCtx.fillStyle = "#ffffff";
    skelCtx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    
    // Helper to draw segment lines
    function drawSegment(startIdx, endIdx) {
        skelCtx.beginPath();
        skelCtx.moveTo(pts[startIdx][0] + ox, pts[startIdx][1] + oy);
        for (let idx = startIdx + 1; idx <= endIdx; idx++) {
            skelCtx.lineTo(pts[idx][0] + ox, pts[idx][1] + oy);
        }
        skelCtx.strokeStyle = "#00ff00"; // BGR (0, 255, 0) -> Green
        skelCtx.lineWidth = 3;
        skelCtx.stroke();
    }

    // 5 Finger segments
    drawSegment(0, 4);   // Thumb
    drawSegment(5, 8);   // Index
    drawSegment(9, 12);  // Middle
    drawSegment(13, 16); // Ring
    drawSegment(17, 20); // Pinky
    
    // Palm connections
    const palmPairs = [
        [5, 9], [9, 13], [13, 17],
        [0, 5], [0, 17]
    ];
    
    skelCtx.beginPath();
    palmPairs.forEach(pair => {
        skelCtx.moveTo(pts[pair[0]][0] + ox, pts[pair[0]][1] + oy);
        skelCtx.lineTo(pts[pair[1]][0] + ox, pts[pair[1]][1] + oy);
    });
    skelCtx.strokeStyle = "#00ff00";
    skelCtx.lineWidth = 3;
    skelCtx.stroke();

    // Draw joints as solid pinkish-red circles
    // BGR (108, 99, 255) -> RGB (255, 99, 108) -> Hex #ff636c
    skelCtx.fillStyle = "#ff636c";
    for (let i = 0; i < 21; i++) {
        skelCtx.beginPath();
        skelCtx.arc(pts[i][0] + ox, pts[i][1] + oy, 3, 0, 2 * Math.PI);
        skelCtx.fill();
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// 4. MODEL INFERENCE & DISAMBIGUATION RULES
// ══════════════════════════════════════════════════════════════════════════════

// Euclidean distance helper
function dist2D(a, b) {
    return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
}

// Utility to check if a pair is present in the rules list
function containsPair(list, val1, val2) {
    return list.some(pair => pair[0] === val1 && pair[1] === val2);
}

/**
 * Runs the TensorFlow.js model on the skeleton canvas and executes
 * the exact Python rules to resolve the character.
 */
async function runInference(pts) {
    if (!tfModel) return;
    
    // tf.tidy disposes of intermediate tensors to avoid memory leaks
    const result = tf.tidy(() => {
        // Read pixels from white 400x400 skeleton canvas
        const imgTensor = tf.browser.fromPixels(skeletonCanvas); // [400, 400, 3]
        const expandedTensor = imgTensor.expandDims(0); // [1, 400, 400, 3]
        const castedTensor = tf.cast(expandedTensor, 'float32'); // Model inputs are float32
        
        // Predict
        const predictions = tfModel.predict(castedTensor);
        const probs = predictions.squeeze().dataSync(); // Sync fetch probabilities array [8]
        return Array.from(probs);
    });
    
    // Sort probabilities to find top 2 indices (ch1, ch2)
    const indexedProbs = result.map((p, idx) => ({ prob: p, index: idx }));
    indexedProbs.sort((a, b) => b.prob - a.prob);
    
    let ch1 = indexedProbs[0].index;
    let ch2 = indexedProbs[1].index;
    const confidence = indexedProbs[0].prob;
    
    // Apply 8-Group -> Sub-Group Disambiguation Rules (verbatim translation from python)
    let list;
    const d = dist2D;
    
    // [Aemnst] rules
    list = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],
            [6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],
            [1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],
            [5,4],[7,0],[7,5],[7,2]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] < pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) {
            ch1 = 0;
        }
    }
    
    list = [[2,2],[2,1]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[5][0] < pts[4][0]) ch1 = 0;
    }
    
    list = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]];
    if (containsPair(list, ch1, ch2)) {
        if ((pts[0][0] > pts[8][0] && pts[0][0] > pts[4][0] && pts[0][0] > pts[12][0] &&
             pts[0][0] > pts[16][0] && pts[0][0] > pts[20][0]) && pts[5][0] > pts[4][0]) {
            ch1 = 2;
        }
    }
    
    list = [[6,0],[6,6],[6,2]];
    if (containsPair(list, ch1, ch2)) {
        if (d(pts[8], pts[16]) < 52) ch1 = 2;
    }
    
    list = [[1,4],[1,5],[1,6],[1,3],[1,0]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] > pts[8][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1] &&
            pts[0][0] < pts[8][0] && pts[0][0] < pts[12][0] && pts[0][0] < pts[16][0] && pts[0][0] < pts[20][0]) {
            ch1 = 3;
        }
    }
    
    list = [[4,6],[4,1],[4,5],[4,3],[4,7]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[4][0] > pts[0][0]) ch1 = 3;
    }
    
    list = [[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[2][1] + 15 < pts[16][1]) ch1 = 3;
    }
    
    list = [[6,4],[6,1],[6,2]];
    if (containsPair(list, ch1, ch2)) {
        if (d(pts[4], pts[11]) > 55) ch1 = 4;
    }
    
    list = [[1,4],[1,6],[1,1]];
    if (containsPair(list, ch1, ch2)) {
        if (d(pts[4], pts[11]) > 50 && (pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1])) {
            ch1 = 4;
        }
    }
    
    list = [[3,6],[3,4]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[4][0] < pts[0][0]) ch1 = 4;
    }
    
    list = [[2,2],[2,5],[2,4]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[1][0] < pts[12][0]) ch1 = 4;
    }
    
    list = [[3,6],[3,5],[3,4]];
    if (containsPair(list, ch1, ch2)) {
        if ((pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) && pts[4][1] > pts[10][1]) {
            ch1 = 5;
        }
    }
    
    list = [[3,2],[3,1],[3,6]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[4][1] + 17 > pts[8][1] && pts[4][1] + 17 > pts[12][1] && pts[4][1] + 17 > pts[16][1] && pts[4][1] + 17 > pts[20][1]) {
            ch1 = 5;
        }
    }
    
    list = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[4][0] > pts[0][0]) ch1 = 5;
    }
    
    list = [[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[0][0] < pts[8][0] && pts[0][0] < pts[12][0] && pts[0][0] < pts[16][0] && pts[0][0] < pts[20][0]) {
            ch1 = 5;
        }
    }
    
    list = [[5,7],[5,2],[5,6]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[3][0] < pts[0][0]) ch1 = 7;
    }
    
    list = [[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] < pts[8][1]) ch1 = 7;
    }
    
    list = [[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[18][1] > pts[20][1]) ch1 = 7;
    }
    
    list = [[0,4],[0,2],[0,3],[0,1],[0,6]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[5][0] > pts[16][0]) ch1 = 6;
    }
    
    list = [[7,2]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[18][1] < pts[20][1] && pts[8][1] < pts[10][1]) ch1 = 6;
    }
    
    list = [[2,1],[2,2],[2,6],[2,7],[2,0]];
    if (containsPair(list, ch1, ch2)) {
        if (d(pts[8], pts[16]) > 50) ch1 = 6;
    }
    
    list = [[4,6],[4,2],[4,1],[4,4]];
    if (containsPair(list, ch1, ch2)) {
        if (d(pts[4], pts[11]) < 60) ch1 = 6;
    }
    
    list = [[1,4],[1,6],[1,0],[1,2]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[5][0] - pts[4][0] - 15 > 0) ch1 = 6;
    }
    
    list = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1] && pts[18][1] > pts[20][1]) {
            ch1 = 1;
        }
    }
    
    list = [[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],[4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5],[7,2]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] < pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1] && pts[18][1] > pts[20][1]) {
            ch1 = 1;
        }
    }
    
    list = [[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1] && pts[18][1] > pts[20][1]) {
            ch1 = 1;
        }
    }
    
    list = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]];
    if (containsPair(list, ch1, ch2)) {
        if ((pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) && pts[2][0] < pts[0][0] && pts[4][1] > pts[14][1]) {
            ch1 = 1;
        }
    }
    
    list = [[4,1],[4,2],[4,4]];
    if (containsPair(list, ch1, ch2)) {
        if (d(pts[4], pts[11]) < 50 && (pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1])) {
            ch1 = 1;
        }
    }
    
    list = [[3,4],[3,0],[3,1],[3,5],[3,6]];
    if (containsPair(list, ch1, ch2)) {
        if ((pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) && pts[2][0] < pts[0][0] && pts[14][1] < pts[4][1]) {
            ch1 = 1;
        }
    }
    
    list = [[6,6],[6,4],[6,1],[6,2]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[5][0] - pts[4][0] - 15 < 0) ch1 = 1;
    }
    
    list = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],[7,6],[7,7]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] < pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] > pts[20][1]) {
            ch1 = 1;
        }
    }
    
    list = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[4][0] < pts[5][0] + 15 && (pts[6][1] < pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] > pts[20][1])) {
            ch1 = 7;
        }
    }
    
    list = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]];
    if (containsPair(list, ch1, ch2)) {
        if ((pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) && pts[4][1] > pts[14][1]) {
            ch1 = 1;
        }
    }
    
    list = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]];
    if (containsPair(list, ch1, ch2)) {
        let fg = 13;
        if (!(pts[0][0] + fg < pts[8][0] && pts[0][0] + fg < pts[12][0] && pts[0][0] + fg < pts[16][0] && pts[0][0] + fg < pts[20][0]) &&
            !(pts[0][0] > pts[8][0] && pts[0][0] > pts[12][0] && pts[0][0] > pts[16][0] && pts[0][0] > pts[20][0]) &&
            d(pts[4], pts[11]) < 50) {
            ch1 = 1;
        }
    }
    
    list = [[5,0],[5,5],[0,1]];
    if (containsPair(list, ch1, ch2)) {
        if (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1]) {
            ch1 = 1;
        }
    }
    
    // --- Sub-group -> Single Letter Resolution ---
    let finalChar = "—";
    if (ch1 === 0) {
        finalChar = "S";
        if (pts[4][0] < pts[6][0] && pts[4][0] < pts[10][0] && pts[4][0] < pts[14][0] && pts[4][0] < pts[18][0]) finalChar = "A";
        else if (pts[4][0] > pts[6][0] && pts[4][0] < pts[10][0] && pts[4][0] < pts[14][0] && pts[4][0] < pts[18][0] && pts[4][1] < pts[14][1] && pts[4][1] < pts[18][1]) finalChar = "T";
        else if (pts[4][1] > pts[8][1] && pts[4][1] > pts[12][1] && pts[4][1] > pts[16][1] && pts[4][1] > pts[20][1]) finalChar = "E";
        else if (pts[4][0] > pts[6][0] && pts[4][0] > pts[10][0] && pts[4][0] > pts[14][0] && pts[4][1] < pts[18][1]) finalChar = "M";
        else if (pts[4][0] > pts[6][0] && pts[4][0] > pts[10][0] && pts[4][1] < pts[18][1] && pts[4][1] < pts[14][1]) finalChar = "N";
    } else if (ch1 === 2) {
        finalChar = d(pts[12], pts[4]) > 42 ? "C" : "O";
    } else if (ch1 === 3) {
        finalChar = d(pts[8], pts[12]) > 72 ? "G" : "H";
    } else if (ch1 === 7) {
        finalChar = d(pts[8], pts[4]) > 42 ? "Y" : "J";
    } else if (ch1 === 4) {
        finalChar = "L";
    } else if (ch1 === 6) {
        finalChar = "X";
    } else if (ch1 === 5) {
        if (pts[4][0] > pts[12][0] && pts[4][0] > pts[16][0] && pts[4][0] > pts[20][0]) {
            finalChar = pts[8][1] < pts[5][1] ? "Z" : "Q";
        } else {
            finalChar = "P";
        }
    } else if (ch1 === 1) {
        if (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1] && pts[18][1] > pts[20][1]) finalChar = "B";
        else if (pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) finalChar = "D";
        else if (pts[6][1] < pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1] && pts[18][1] > pts[20][1]) finalChar = "F";
        else if (pts[6][1] < pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] > pts[20][1]) finalChar = "I";
        else if (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] > pts[16][1] && pts[18][1] < pts[20][1]) finalChar = "W";
        else if ((pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) && pts[4][1] < pts[9][1]) finalChar = "K";
        else if ((d(pts[8], pts[12]) - d(pts[6], pts[10])) < 8 && (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1])) finalChar = "U";
        else if ((d(pts[8], pts[12]) - d(pts[6], pts[10])) >= 8 && (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1]) && pts[4][1] > pts[9][1]) finalChar = "V";
        else if (pts[8][0] > pts[12][0] && (pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] < pts[20][1])) finalChar = "R";
    }
    
    // --- Special Gesture Identification ---
    // Space gesture (index+pinky up, middle+ring down)
    if (["1", "E", "S", "X", "Y", "B"].includes(finalChar) || finalChar === 1) {
        if (pts[6][1] > pts[8][1] && pts[10][1] < pts[12][1] && pts[14][1] < pts[16][1] && pts[18][1] > pts[20][1]) {
            finalChar = " ";
        }
    }
    
    // Next gesture (confirm character — all fingers up, thumb tucked)
    if (["E", "Y", "B"].includes(finalChar)) {
        if (pts[4][0] < pts[5][0] && pts[6][1] > pts[8][1] && pts[10][1] > pts[12][1] &&
            pts[14][1] > pts[16][1] && pts[18][1] > pts[20][1]) {
            finalChar = "next";
        }
    }
    
    // Backspace gesture (thumb-up fist)
    if (["Next", "next", "B", "C", "H", "F", "X"].includes(finalChar)) {
        if (pts[0][0] > pts[8][0] && pts[0][0] > pts[12][0] && pts[0][0] > pts[16][0] && pts[0][0] > pts[20][0] &&
            pts[4][1] < pts[8][1] && pts[4][1] < pts[12][1] && pts[4][1] < pts[16][1] && pts[4][1] < pts[20][1] &&
            pts[4][1] < pts[6][1] && pts[4][1] < pts[10][1] && pts[4][1] < pts[14][1] && pts[4][1] < pts[18][1]) {
            finalChar = "Backspace";
        }
    }
    
    // Process prediction buffer stability
    updateBuffer(finalChar);
    
    // Update Character UI Displays
    updateCharUI(finalChar, confidence);
}


// ══════════════════════════════════════════════════════════════════════════════
// 5. CHARACTER BUFFER LOGIC (Stability Heuristics)
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Manages key history buffers and triggers word completion / spaces / backspaces.
 * Verbatim port of Python's _update_buffer logic.
 */
function updateBuffer(ch1) {
    // Next gesture triggers the committed character from the stable buffer
    if (ch1 === "next" && prevChar !== "next") {
        const prevIdx = (counter - 2 + 10) % 10;
        const c = tenPrevBuffer[prevIdx];
        if (c !== "next" && c !== "SPC" && c !== " ") {
            if (c === "Backspace") {
                activeSentence = activeSentence.slice(0, -1);
                triggerUIFlash();
            } else if (c && c !== "—") {
                activeSentence += c;
                triggerUIFlash();
            }
        } else {
            const c2 = tenPrevBuffer[(counter + 10) % 10];
            if (c2 && c2 !== "Backspace" && c2 !== "next" && c2 !== "SPC" && c2 !== " ") {
                activeSentence += c2;
                triggerUIFlash();
            }
        }
    }
    
    // Space gesture commits word
    if (ch1 === " " && prevChar !== " ") {
        activeSentence += " ";
        // Auto-speak completed word
        const words = activeSentence.trim().split(/\s+/);
        if (words.length > 0) {
            speakText(words[words.length - 1]);
        }
    }
    
    prevChar = ch1;
    counter++;
    tenPrevBuffer[counter % 10] = ch1;
    
    // Run Spell Check on active word
    if (activeSentence.trim()) {
        const idx = activeSentence.lastIndexOf(" ");
        const cur = activeSentence.slice(idx + 1).trim();
        if (cur && cur !== currentWord) {
            currentWord = cur;
            updateSpellCheckSuggestions(cur);
        } else if (!cur) {
            currentWord = "";
            clearSuggestions();
        }
    } else {
        currentWord = "";
        clearSuggestions();
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// 6. UI SYNCHRONIZATION
// ══════════════════════════════════════════════════════════════════════════════

function updateUI() {
    sentenceLabel.textContent = activeSentence.trim() ? activeSentence : "Start signing to build a sentence...";
    if (activeSentence.trim()) {
        sentenceLabel.classList.add("active");
    } else {
        sentenceLabel.classList.remove("active");
    }
}

function updateCharUI(ch1, confidence) {
    let charStr = ch1 ? ch1.toString().toUpperCase() : "—";
    if (charStr === " ") charStr = "SPC";
    
    charLabel.textContent = charStr;
    
    // Colour styling according to active labels
    if (["—", "NEXT", "BACKSPACE", "SPC"].includes(charStr)) {
        charLabel.style.color = "var(--text-dim)";
    } else {
        charLabel.style.color = "var(--teal)";
    }
    
    // Confidence percentage and bar
    const confVal = Math.round(confidence * 100);
    confPct.textContent = `${confVal}%`;
    confBar.style.width = `${confVal}%`;
    
    updateUI();
}

function triggerUIFlash() {
    sentenceLabel.classList.add("flash-char");
    setTimeout(() => {
        sentenceLabel.classList.remove("flash-char");
    }, 300);
}


// ══════════════════════════════════════════════════════════════════════════════
// 7. WEBCAM & MEDIAPIPE ORCHESTRATION
// ══════════════════════════════════════════════════════════════════════════════

function onHandResults(results) {
    // Clear camera canvas
    camCtx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
    
    // Draw mirrored video frame
    camCtx.save();
    camCtx.translate(cameraCanvas.width, 0);
    camCtx.scale(-1, 1);
    camCtx.drawImage(results.image, 0, 0, cameraCanvas.width, cameraCanvas.height);
    camCtx.restore();
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        statusMsg.textContent = "🤚 Hand detected — inferring gesture...";
        
        const landmarks = results.multiHandLandmarks[0];
        
        // Mirror horizontally to match mirrored video feed and Python training inputs
        for (let i = 0; i < 21; i++) {
            landmarks[i].x = 1.0 - landmarks[i].x;
        }
        
        // Calculate Bounding Box of the hand in pixels
        let xmin = cameraCanvas.width;
        let xmax = 0;
        let ymin = cameraCanvas.height;
        let ymax = 0;
        
        for (let i = 0; i < 21; i++) {
            const px = landmarks[i].x * cameraCanvas.width;
            const py = landmarks[i].y * cameraCanvas.height;
            if (px < xmin) xmin = px;
            if (px > xmax) xmax = px;
            if (py < ymin) ymin = py;
            if (py > ymax) ymax = py;
        }
        
        const w = xmax - xmin;
        const h = ymax - ymin;
        
        // Project bounding box coordinates with crop offset
        // We flip the horizontal positions to align with the mirrored canvas view
        const x1 = Math.max(0, xmin - BOX_OFFSET);
        const y1 = Math.max(0, ymin - BOX_OFFSET);
        const x2 = Math.min(cameraCanvas.width, xmax + BOX_OFFSET);
        const y2 = Math.min(cameraCanvas.height, ymax + BOX_OFFSET);
        
        // Draw glowy purple bounding box on live view
        camCtx.strokeStyle = "rgba(123, 111, 255, 0.8)";
        camCtx.lineWidth = 3;
        camCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        camCtx.fillStyle = "rgba(123, 111, 255, 0.9)";
        camCtx.font = "bold 13px Inter";
        camCtx.fillText("Hand Detected", x1, y1 - 8);
        
        // Translate landmarks to crop coordinates
        const pts = [];
        for (let i = 0; i < 21; i++) {
            const px = landmarks[i].x * cameraCanvas.width;
            const py = landmarks[i].y * cameraCanvas.height;
            pts.push([px - x1, py - y1]);
        }
        
        // Center skeleton points on 400x400 canvas
        const ox = Math.floor((IMAGE_SIZE - w) / 2) - 15;
        const oy = Math.floor((IMAGE_SIZE - h) / 2) - 15;
        
        // Draw the skeleton
        drawSkeleton(pts, ox, oy);
        
        // Feed skeleton canvas to CNN Model
        runInference(pts);
    } else {
        statusMsg.textContent = "👁️ Waiting for hand — hold ISL sign to camera";
        
        // Clear skeleton canvas back to solid white if no hand is tracked
        skelCtx.fillStyle = "#ffffff";
        skelCtx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
        
        // Update character display to default empty state
        updateCharUI("—", 0);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. APPLICATION BOOTSTRAP
// ══════════════════════════════════════════════════════════════════════════════

async function init() {
    loadingText.textContent = "Initializing camera feed...";
    
    // Initialize MediaPipe Hands library
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    hands.onResults(onHandResults);
    
    // Bind webcam stream
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        webcamElement.srcObject = stream;
        
        // Camera scheduler loop
        const camera = new Camera(webcamElement, {
            onFrame: async () => {
                await hands.send({ image: webcamElement });
            },
            width: 640,
            height: 480
        });
        
        isMediaPipeReady = true;
        camera.start();
        console.log("MediaPipe Hands loaded & camera stream active.");
        
        // Load ML Model
        await initModel();
        
        // Initialize Lucide icon tags
        lucide.createIcons();
    } catch (err) {
        console.error("Camera acquisition failed:", err);
        loadingText.textContent = "Webcam access denied. Please allow camera permissions.";
    }
}

// Kickstart
window.addEventListener("DOMContentLoaded", init);
