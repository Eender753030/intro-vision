/* ==========================================================================
   EmoVision Premium Javascript Controller (Browser Web Serial System)
   ========================================================================== */

// DOM Elements Selection
const btnConnect = document.getElementById('btn-connect');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const baudrateSelect = document.getElementById('baudrate');

// Canvases contexts
const canvasMain = document.getElementById('canvas-main');
const ctxMain = canvasMain.getContext('2d');
const canvasFace = document.getElementById('canvas-face');
const ctxFace = canvasFace.getContext('2d');

// Bounding box telemetry elements
const valX = document.getElementById('val-x');
const valY = document.getElementById('val-y');
const valW = document.getElementById('val-w');
const valH = document.getElementById('val-h');

// Decision & Telemetries Elements
const topName = document.getElementById('top-emotion-name');
const topConf = document.getElementById('top-emotion-conf');
const topBubble = document.getElementById('top-emotion-bubble');
const statLatency = document.getElementById('stat-latency');
const statFaceLatency = document.getElementById('stat-face-latency');
const statFps = document.getElementById('stat-fps');
const statSram = document.getElementById('stat-sram');
const statPsram = document.getElementById('stat-psram');

// Log Console
const logConsole = document.getElementById('log-console');
const chkAutoscroll = document.getElementById('chk-autoscroll');
const btnClearLog = document.getElementById('btn-clear-log');

// Alarm overlay
const alarmOverlay = document.getElementById('haptic-alarm');
const alarmDetail = document.getElementById('alarm-detail');

// Debounce telemetry elements
const dbSustainCard = document.getElementById('db-state-sustain');
const dbCooldownCard = document.getElementById('db-state-cooldown');
const dbIdleCard = document.getElementById('db-state-idle');
const dbTimeVal = document.getElementById('db-time-val');
const dbBarFill = document.getElementById('db-bar-fill');

// State Variables
let serialPort = null;
let serialReader = null;
let keepReading = false;
let inputBuffer = "";
let inJpegMode = false;
let jpegHexLines = [];

// Coordinates & Drawing
let faceCoords = null; // { x, y, w, h }
let lastFrameImg = null; // Retain last image object to redraw box

// Connection Setup
btnConnect.addEventListener('click', async () => {
    if (serialPort) {
        // Disconnect
        await disconnectSerial();
    } else {
        // Connect
        await connectSerial();
    }
});

btnClearLog.addEventListener('click', () => {
    logConsole.innerHTML = '<div class="log-line system-line">[SYSTEM] 日誌已清空。</div>';
});

// Setup clean starting state on canvas
function resetTelemetryUI() {
    ctxMain.fillStyle = '#030305';
    ctxMain.fillRect(0, 0, canvasMain.width, canvasMain.height);
    
    ctxFace.fillStyle = '#000000';
    ctxFace.fillRect(0, 0, canvasFace.width, canvasFace.height);
    
    // Draw "Scanning" text on sub-canvas
    ctxFace.font = '8px monospace';
    ctxFace.fillStyle = '#545469';
    ctxFace.textAlign = 'center';
    ctxFace.fillText('SCANNING...', 24, 26);

    valX.textContent = '-';
    valY.textContent = '-';
    valW.textContent = '-';
    valH.textContent = '-';

    topName.textContent = '無人臉';
    topConf.textContent = 'Confidence: 0.0%';
    topBubble.className = 'decision-bubble';
    
    statLatency.textContent = '-- ms';
    statFaceLatency.textContent = '-- ms';
    statFps.textContent = '-- FPS';

    // Reset bar distributions
    document.querySelectorAll('.bar-fill').forEach(bar => bar.style.width = '0%');
    document.querySelectorAll('.prob-row .val').forEach(val => val.textContent = '0.0%');
    document.querySelectorAll('.prob-row').forEach(row => row.classList.remove('top-active'));
}
resetTelemetryUI();

// --------------------------------------------------------------------------
// Serial Communication Logic
// --------------------------------------------------------------------------

async function connectSerial() {
    if (!("serial" in navigator)) {
        appendLogLine("[ERROR] 您的瀏覽器不支援 Web Serial API！請使用 Google Chrome 或 Microsoft Edge 開啟本頁面。", "err-line");
        return;
    }

    try {
        statusDot.className = "dot connecting";
        statusText.textContent = "選擇埠口中 (Connecting)";
        
        serialPort = await navigator.serial.requestPort();
        const baudRate = parseInt(baudrateSelect.value, 10);
        
        await serialPort.open({ baudRate });
        
        statusDot.className = "dot connected";
        statusText.textContent = `已連線: ${baudRate} bps`;
        btnConnect.textContent = "🔌 斷開設備 (Disconnect)";
        btnConnect.className = "btn btn-primary btn-connected";
        document.getElementById('no-signal').style.opacity = '0';
        
        appendLogLine(`[SYSTEM] 成功連接至設備！波特率設定為 ${baudRate} bps`, "system-line");
        
        keepReading = true;
        readSerialStream();
    } catch (err) {
        console.error(err);
        statusDot.className = "dot disconnected";
        statusText.textContent = "連線失敗 (Failed)";
        appendLogLine(`[ERROR] 無法打開序列埠：${err.message}`, "err-line");
        serialPort = null;
    }
}

async function disconnectSerial() {
    keepReading = false;
    if (serialReader) {
        try {
            await serialReader.cancel();
        } catch (e) {}
    }
    
    if (serialPort) {
        try {
            await serialPort.close();
        } catch (e) {}
        serialPort = null;
    }
    
    statusDot.className = "dot disconnected";
    statusText.textContent = "已斷開 (Disconnected)";
    btnConnect.textContent = "⚡ 連接設備 (Connect)";
    btnConnect.className = "btn btn-primary";
    document.getElementById('no-signal').style.opacity = '1';
    
    appendLogLine("[SYSTEM] 序列埠已成功關閉斷開。", "system-line");
    resetTelemetryUI();
}

async function readSerialStream() {
    while (serialPort && serialPort.readable && keepReading) {
        serialReader = serialPort.readable.getReader();
        const decoder = new TextDecoder();
        
        try {
            while (true) {
                const { value, done } = await serialReader.read();
                if (done) {
                    break;
                }
                
                // Decode incoming chunk and feed into buffer
                inputBuffer += decoder.decode(value);
                
                // Split buffer by newlines
                let lines = inputBuffer.split("\n");
                // Keep the last incomplete fragment in the buffer
                inputBuffer = lines.pop();
                
                for (let line of lines) {
                    parseSerialLine(line.trim());
                }
            }
        } catch (err) {
            console.error("Read error:", err);
            appendLogLine(`[ERROR] 讀取序列埠時發生異常：${err.message}`, "err-line");
            break;
        } finally {
            serialReader.releaseLock();
            serialReader = null;
        }
    }
    
    if (serialPort && keepReading) {
        // Auto-reconnect or handle unexpected close
        await disconnectSerial();
    }
}

// --------------------------------------------------------------------------
// Core Parsing Engine
// --------------------------------------------------------------------------

function parseSerialLine(line) {
    if (!line) return;

    // JPEG Data Stream Mode check
    if (line.includes("---BEGIN_JPEG---")) {
        inJpegMode = true;
        jpegHexLines = [];
        return;
    }

    if (inJpegMode) {
        if (line.includes("---END_JPEG---")) {
            inJpegMode = false;
            reconstructJpegFrame();
        } else {
            // Avoid adding serial noise
            if (!line.startsWith("[") && !line.startsWith("I (") && !line.startsWith("W (")) {
                jpegHexLines.push(line);
            }
        }
        return;
    }

    // Normal Text Parsing Mode
    // 1. Log lines coloring and output to console
    printConsoleLine(line);

    // 2. Face Box coordinates
    if (line.includes("FACE_BOX:")) {
        const coordsStr = line.split("FACE_BOX:")[1];
        const [x, y, w, h] = coordsStr.split(",").map(Number);
        
        if (x === 0 && y === 0 && w === 0 && h === 0) {
            // Reset coordinates, face disappeared
            faceCoords = null;
            
            // Repaint sub-canvas to black Scanning
            ctxFace.fillStyle = '#000000';
            ctxFace.fillRect(0, 0, canvasFace.width, canvasFace.height);
            ctxFace.font = '8px monospace';
            ctxFace.fillStyle = '#545469';
            ctxFace.textAlign = 'center';
            ctxFace.fillText('SCANNING...', 24, 26);
            
            valX.textContent = '-';
            valY.textContent = '-';
            valW.textContent = '-';
            valH.textContent = '-';
            
            topName.textContent = '無人臉';
            topConf.textContent = 'Confidence: 0.0%';
            topBubble.className = 'decision-bubble';
            
            // Set Debounce to Idle
            setDebounceUI('idle');
        } else {
            faceCoords = { x, y, w, h };
            valX.textContent = x;
            valY.textContent = y;
            valW.textContent = w;
            valH.textContent = h;
            
            // Immediately draw overlay on canvas if we have lastFrameImg
            if (lastFrameImg) {
                redrawMainCanvas();
            }
        }
        return;
    }

    // 3. Inference probabilities vector
    if (line.includes("EMOTION_PROBS:")) {
        const probsStr = line.split("EMOTION_PROBS:")[1];
        const items = probsStr.split(",");
        
        let highestEmotion = "Neutral";
        let highestProb = -1.0;
        
        for (let item of items) {
            const [name, valStr] = item.split(":");
            const probability = parseFloat(valStr);
            const percentage = (probability * 100).toFixed(1);
            
            // Update individual bar elements
            const barFill = document.querySelector(`.prob-row[data-emotion="${name}"] .bar-fill`);
            const valLabel = document.getElementById(`prob-${name}`);
            
            if (barFill) barFill.style.width = `${percentage}%`;
            if (valLabel) valLabel.textContent = `${percentage}%`;
            
            // Track highest
            if (probability > highestProb) {
                highestProb = probability;
                highestEmotion = name;
            }
        }
        
        // Highlight active highest row
        document.querySelectorAll('.prob-row').forEach(row => {
            if (row.getAttribute('data-emotion') === highestEmotion) {
                row.classList.add('top-active');
            } else {
                row.classList.remove('top-active');
            }
        });

        // Update Decision UI Bubble
        topName.textContent = translateEmotion(highestEmotion);
        topConf.textContent = `Confidence: ${(highestProb * 100).toFixed(1)}%`;
        topBubble.className = `decision-bubble active-${highestEmotion.toLowerCase()}`;
        return;
    }

    // 4. Haptic feedback alarms
    if (line.includes("HAPTIC_TRIGGER:")) {
        const [label, code] = line.split("HAPTIC_TRIGGER:")[1].split(",");
        triggerHapticAlarm(label, code);
        return;
    }

    // 5. Debounce sustain metrics
    if (line.includes("[DEBOUNCE] Sustained")) {
        // e.g. [DEBOUNCE] Sustained Surprise: 178 ms / 200 ms
        const parts = line.split("Sustained")[1].split(":");
        const timeStr = parts[1].trim(); // "178 ms / 200 ms"
        const numbers = timeStr.match(/\d+/g).map(Number);
        
        if (numbers.length >= 2) {
            const currentMs = numbers[0];
            const targetMs = numbers[1];
            const percentage = Math.min((currentMs / targetMs) * 100, 100);
            
            setDebounceUI('sustain', timeStr, percentage);
        }
        return;
    }

    if (line.includes("[DEBOUNCE] Transitioned to safe state") || line.includes("Debounce reset")) {
        setDebounceUI('idle');
        return;
    }

    if (line.includes("ACTIVE COOLDOWN")) {
        // e.g. ACTIVE COOLDOWN (Remaining: 4200 ms)
        const matches = line.match(/Remaining:\s*(\d+)\s*ms/);
        if (matches && matches[1]) {
            const remaining = matches[1];
            setDebounceUI('cooldown', `冷卻中: ${remaining} ms / 5000 ms`, (remaining / 5000) * 100);
        }
        return;
    }

    // 6. Telemetries
    if (line.includes("FACE_DETECT_LATENCY:")) {
        const val = parseFloat(line.split("FACE_DETECT_LATENCY:")[1]).toFixed(1);
        statFaceLatency.textContent = `${val} ms`;
        return;
    }

    if (line.includes("CAMERA_FPS:")) {
        const val = parseFloat(line.split("CAMERA_FPS:")[1]).toFixed(1);
        statFps.textContent = `${val} FPS`;
        return;
    }

    if (line.includes(">>> ESP-DL Inference Latency:")) {
        // e.g. >>> ESP-DL Inference Latency: 152.15 ms <<<
        const latency = line.split("Inference Latency:")[1].split("<<<")[0].trim();
        statLatency.textContent = latency;
        return;
    }

    if (line.includes("- Internal SRAM Free:")) {
        const sram = line.split("Internal SRAM Free:")[1].split("Bytes")[0].trim();
        const kb = (parseInt(sram, 10) / 1024).toFixed(1);
        statSram.textContent = `${kb} KB`;
        return;
    }

    if (line.includes("- External PSRAM Free:")) {
        const psram = line.split("External PSRAM Free:")[1].split("Bytes")[0].trim();
        const mb = (parseInt(psram, 10) / 1024 / 1024).toFixed(2);
        statPsram.textContent = `${mb} MB`;
        return;
    }
}

// --------------------------------------------------------------------------
// Video Frames Reconstructor
// --------------------------------------------------------------------------

function reconstructJpegFrame() {
    const hexStr = jpegHexLines.join("").trim();
    if (hexStr.length === 0) return;

    try {
        const len = hexStr.length / 2;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = parseInt(hexStr.substr(i * 2, 2), 16);
        }

        const blob = new Blob([bytes], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        
        img.onload = () => {
            lastFrameImg = img;
            redrawMainCanvas();
            URL.revokeObjectURL(url);
        };
        img.src = url;
    } catch (err) {
        console.error("Hex array mapping failed", err);
    }
}

// Redraw camera frame and overlay boxes + crop AI sub-canvas
function redrawMainCanvas() {
    if (!lastFrameImg) return;
    
    // 1. Draw full image on main canvas
    ctxMain.drawImage(lastFrameImg, 0, 0, canvasMain.width, canvasMain.height);
    
    // 2. Draw active green bounding box tracking frame
    if (faceCoords) {
        const { x, y, w, h } = faceCoords;
        
        // Draw cybernetic tracking rect
        ctxMain.strokeStyle = '#0df07e'; // Neon Green
        ctxMain.lineWidth = 2;
        ctxMain.shadowColor = 'rgba(13, 240, 126, 0.4)';
        ctxMain.shadowBlur = 8;
        ctxMain.strokeRect(x, y, w, h);
        
        // Reset shadows for clean graphics
        ctxMain.shadowBlur = 0;
        
        // Draw dynamic coordinate anchors text
        ctxMain.fillStyle = '#0df07e';
        ctxMain.font = '9px monospace';
        ctxMain.fillText(`ID: FACE [${x},${y},${w}x${h}]`, x, y > 15 ? y - 5 : y + h + 12);

        // 3. Dynamic Cropping from the same frame into AI Canvas
        cropAndRenderGrayscale(lastFrameImg, x, y, w, h);
    }
}

function cropAndRenderGrayscale(img, x, y, w, h) {
    // 1. Bounding clamping limits check
    const cleanX = Math.max(0, x);
    const cleanY = Math.max(0, y);
    const cleanW = Math.min(w, img.width - cleanX);
    const cleanH = Math.min(h, img.height - cleanY);

    if (cleanW <= 0 || cleanH <= 0) return;

    // Create an intermediate canvas for cropping
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cleanW;
    tempCanvas.height = cleanH;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw cropped region from image onto temp canvas
    tempCtx.drawImage(img, cleanX, cleanY, cleanW, cleanH, 0, 0, cleanW, cleanH);

    // Convert temp canvas to Grayscale
    const imgData = tempCtx.getImageData(0, 0, cleanW, cleanH);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
        // Fast integer grayscale luma weights
        const gray = (data[i] * 77 + data[i+1] * 150 + data[i+2] * 29) >> 8;
        data[i] = gray;     // R
        data[i+1] = gray;   // G
        data[i+2] = gray;   // B
    }
    tempCtx.putImageData(imgData, 0, 0);

    // Render it resized (48x48) on sub-canvas
    ctxFace.fillStyle = '#000000';
    ctxFace.fillRect(0, 0, canvasFace.width, canvasFace.height);
    
    // Use canvas scaling (bilinear scaling is fine because image-rendering: pixelated handles visual sizing)
    ctxFace.drawImage(tempCanvas, 0, 0, tempCanvas.width, tempCanvas.height, 0, 0, canvasFace.width, canvasFace.height);
}

// --------------------------------------------------------------------------
// UI Interaction and Formatting Helpers
// --------------------------------------------------------------------------

function translateEmotion(emotion) {
    const dict = {
        "Happiness": "😊 喜悅 (Happiness)",
        "Surprise": "😲 驚訝 (Surprise)",
        "Neutral": "😐 中性 (Neutral)",
        "Sadness": "😢 悲傷 (Sadness)",
        "Anger": "😡 憤怒 (Anger)",
        "Fear": "😨 恐懼 (Fear)",
        "Disgust": "🤢 厭惡 (Disgust)"
    };
    return dict[emotion] || emotion;
}

function setDebounceUI(state, timeText = "0 ms / 200 ms", fillPercent = 0) {
    dbSustainCard.className = "db-state-item";
    dbCooldownCard.className = "db-state-item";
    dbIdleCard.className = "db-state-item";

    if (state === 'idle') {
        dbIdleCard.className = "db-state-item active";
        dbTimeVal.textContent = "0 ms / 200 ms";
        dbBarFill.style.width = "0%";
    } else if (state === 'sustain') {
        dbSustainCard.className = "db-state-item active";
        dbTimeVal.textContent = timeText;
        dbBarFill.style.width = `${fillPercent}%`;
    } else if (state === 'cooldown') {
        dbCooldownCard.className = "db-state-item active";
        dbTimeVal.textContent = timeText;
        dbBarFill.style.width = `${fillPercent}%`;
    }
}

let hapticAlarmTimeout = null;
function triggerHapticAlarm(label, code) {
    // 1. Show flashing alarm overlay on DOM
    alarmDetail.textContent = getHapticPatternDescription(label);
    alarmOverlay.classList.add('triggered');
    
    // Optional Web Vibrate
    if ("vibrate" in navigator) {
        navigator.vibrate([200, 100, 200]);
    }

    // 2. Schedule automatic fade-out
    if (hapticAlarmTimeout) {
        clearTimeout(hapticAlarmTimeout);
    }
    hapticAlarmTimeout = setTimeout(() => {
        alarmOverlay.classList.remove('triggered');
        hapticAlarmTimeout = null;
    }, 1800);
}

function getHapticPatternDescription(label) {
    const patterns = {
        "Sadness": "😢 悲傷觸發：緩慢平滑漸弱 100% 到 0% (Effect 70, 1000ms)",
        "Anger": "😡 憤怒觸發：雙重長震動強嗡嗡聲 (Effect 14 x2, 1500ms)",
        "Fear": "😨 恐懼觸發：三波真實模擬心跳震感 (Effect 1 + 3 + Delays, 1000ms)",
        "Disgust": "🤢 厭惡觸發：連續長 7 倍粗糙不適摩擦震感 (Effect 22 x7, 2000ms)",
        "Surprise": "😲 驚訝觸發：強烈四連發單擊 (Effect 1 x4 + Delays, 400ms, 體感 3 連點)"
    };
    return patterns[label] || `${label}情緒觸發：標準觸覺震動 (DRV2605L)`;
}

// Color coding log terminal
function printConsoleLine(line) {
    // Filter out high-frequency raw telemetry lines from the visual console to prevent DOM lag and free up CPU
    if (line.includes("FACE_DETECT_LATENCY:") || line.includes("CAMERA_FPS:") || line.includes("EMOTION_PROBS:") || line.includes("FACE_BOX:")) {
        return;
    }

    let logClass = "log-line";
    if (line.includes("E (") || line.includes("[ERROR]") || line.includes("[DEBOUNCE] New Warning")) {
        logClass += " err-line";
    } else if (line.includes("W (") || line.includes("[WARN]")) {
        logClass += " warn-line";
    } else if (line.includes("I (") && (line.includes("MAIN:") || line.includes("HAPTICS:"))) {
        logClass += " info-line";
    } else if (line.includes("D (") || line.includes("[DEBUG]")) {
        logClass += " debug-line";
    } else if (line.includes("[DEBOUNCE] >>> SUCCESS:") || line.includes("[ACTUATOR]")) {
        logClass += " system-line";
    }

    appendLogLine(line, logClass);
}

function appendLogLine(text, className) {
    const lineDiv = document.createElement('div');
    lineDiv.className = className;
    lineDiv.textContent = text;
    logConsole.appendChild(lineDiv);
    
    // Auto Scroll if toggled
    if (chkAutoscroll.checked) {
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    // Cap output at 300 lines to prevent memory bloating
    if (logConsole.childElementCount > 300) {
        logConsole.removeChild(logConsole.firstChild);
    }
}
