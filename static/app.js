// Apex Bank Voice Governance Frontend Logic
let ws = null;
let audioContext = null;
let micStream = null;
let scriptProcessor = null;
let analyser = null;
let isConnected = false;
let isMuted = false;
let injectionCount = 0;

// UI Elements
const btnConnect = document.getElementById('btnConnect');
const btnMute = document.getElementById('btnMute');
const connStatus = document.getElementById('connStatus');
const fsmBadge = document.getElementById('fsmBadge');
const transcriptBox = document.getElementById('transcriptBox');
const contextFeed = document.getElementById('contextFeed');
const auditLogBox = document.getElementById('auditLogBox');
const injectionCountBadge = document.getElementById('injectionCountBadge');
const visualizerOverlay = document.getElementById('visualizerOverlay');
const canvas = document.getElementById('waveformCanvas');
const canvasCtx = canvas.getContext('2d');

// Simulation Buttons
const btnSimBereavement = document.getElementById('btnSimBereavement');
const btnSimAdvice = document.getElementById('btnSimAdvice');
const btnSimKillSwitch = document.getElementById('btnSimKillSwitch');

// 1. WebSocket Session Management
function toggleSession() {
    if (isConnected) {
        disconnectSession();
    } else {
        connectSession();
    }
}

let pingTimer = null;

async function connectSession() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    connStatus.textContent = 'Connecting...';
    connStatus.className = 'status-pill';

    ws.onopen = async () => {
        isConnected = true;
        connStatus.textContent = 'Active (Live Streaming)';
        connStatus.className = 'status-pill active';
        btnConnect.innerHTML = `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg> Disconnect Voice Session`;
        btnConnect.className = 'btn btn-secondary';
        btnMute.disabled = false;
        visualizerOverlay.style.display = 'none';

        // Start Keepalive Heartbeat ping every 10 seconds to keep WebSocket connection active
        if (pingTimer) clearInterval(pingTimer);
        pingTimer = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'ping' }));
            }
        }, 10000);

        // Initialize Microphone Audio Capture
        await startMicrophone();
        appendAuditLog('[System] Voice session connected to Gemini Live API on Vertex AI.');
    };

    ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            // JSON Control Event
            const msg = JSON.parse(event.data);
            handleServerEvent(msg);
        } else if (event.data instanceof ArrayBuffer) {
            // Incoming PCM Audio from Gemini (24kHz PCM)
            playAudioChunk(event.data);
        }
    };

    ws.onclose = () => {
        disconnectSession();
    };

    ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
        disconnectSession();
    };
}

function disconnectSession() {
    isConnected = false;
    if (pingTimer) {
        clearInterval(pingTimer);
        pingTimer = null;
    }
    stopMicrophone();

    if (ws) {
        ws.close();
        ws = null;
    }

    connStatus.textContent = 'Disconnected';
    connStatus.className = 'status-pill disconnected';
    btnConnect.innerHTML = `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg> Start Voice Session`;
    btnConnect.className = 'btn btn-primary';
    btnMute.disabled = true;
    visualizerOverlay.style.display = 'block';
    appendAuditLog('[System] Voice session disconnected.');
}

// Helper: Resample Float32 array to 16kHz Int16 PCM array for Gemini
function resampleAndToInt16(float32Array, inSampleRate, outSampleRate = 16000) {
    if (inSampleRate === outSampleRate) {
        const int16 = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16;
    }

    const ratio = inSampleRate / outSampleRate;
    const newLength = Math.round(float32Array.length / ratio);
    const int16 = new Int16Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const originIndex = i * ratio;
        const indexFloor = Math.floor(originIndex);
        const indexCeil = Math.min(float32Array.length - 1, indexFloor + 1);
        const interpolation = originIndex - indexFloor;

        const sample = (1 - interpolation) * float32Array[indexFloor] + interpolation * float32Array[indexCeil];
        const s = Math.max(-1, Math.min(1, sample));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

// 2. Microphone Capture & WebAudio Visualizer
async function startMicrophone() {
    try {
        // Initialize AudioContext configured for 16kHz audio
        if (!audioContext) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            try {
                audioContext = new AudioContextClass({ sampleRate: 16000 });
            } catch (e) {
                audioContext = new AudioContextClass();
            }
        }

        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        // Check if mediaDevices API is available (null when HTTP is unsecure or blocked)
        if (!navigator || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const err = new Error("navigator.mediaDevices is null (Insecure HTTP context). Please use https:// or SSH port forwarding.");
            err.name = "InsecureContextError";
            throw err;
        }

        micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });

        const source = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;

        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        let chunkCount = 0;
        scriptProcessor.onaudioprocess = (e) => {
            if (!isConnected || isMuted || !ws || ws.readyState !== WebSocket.OPEN) return;

            const float32Array = e.inputBuffer.getChannelData(0);
            const currentSampleRate = audioContext.sampleRate;
            
            // Resample Float32 PCM to 16kHz Int16 PCM Mono for Gemini Live API
            const int16Array = resampleAndToInt16(float32Array, currentSampleRate, 16000);

            if (int16Array.length > 0) {
                ws.send(int16Array.buffer);
                chunkCount++;
                if (chunkCount === 1) {
                    appendAuditLog(`[Mic Active] Live microphone audio streaming to Gemini (Native: ${currentSampleRate}Hz -> Resampled: 16000Hz).`);
                }
            }
        };

        // Create zero-gain node to route scriptProcessor without playing mic echo back to speakers
        const silenceGain = audioContext.createGain();
        silenceGain.gain.value = 0;

        source.connect(analyser);
        analyser.connect(scriptProcessor);
        scriptProcessor.connect(silenceGain);
        silenceGain.connect(audioContext.destination);

        drawWaveform();
    } catch (err) {
        console.error('Microphone Access Error:', err);
        const errMsg = `${err.name}: ${err.message}`;
        appendAuditLog(`[Mic Warning] ${errMsg}`);
        
        if (err.name === 'InsecureContextError' || !navigator.mediaDevices) {
            alert(`Microphone Access Blocked (${err.name})\n\nChrome sets 'navigator.mediaDevices' to null on plain HTTP remote connections.\n\nSOLUTION:\nAccess the site via HTTPS (https://${window.location.hostname}:8000) or SSH port forwarding.\n\nNOTE: You can still test all Gemini functions using the Quick Test Triggers on the right!`);
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            alert(`No microphone hardware detected on this Chrometop session (${err.name}).\n\nThe WebSocket session is ACTIVE! You can test all voice agent features, RAG context injections, and state changes using the Quick Test Triggers on the right.`);
        } else {
            alert(`Microphone Error (${err.name}): ${err.message}\n\nPlease verify browser site permissions or use the Quick Test Triggers.`);
        }
    }
}

function stopMicrophone() {
    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
}

function toggleMute() {
    isMuted = !isMuted;
    btnMute.textContent = isMuted ? 'Unmute Mic' : 'Mute Mic';
    btnMute.style.borderColor = isMuted ? 'var(--accent-red)' : 'var(--card-border)';
}

// 3. Audio Waveform Canvas Drawing
function drawWaveform() {
    if (!analyser) return;

    requestAnimationFrame(drawWaveform);
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = '#090E17';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height;

        const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#00F2FE');
        gradient.addColorStop(1, '#4FACFE');

        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
    }
}

// 4. Audio Playback (24kHz PCM from Gemini)
let nextStartTime = 0;
let audioEndTime = 0;

async function playAudioChunk(arrayBuffer) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    const int16Array = new Int16Array(arrayBuffer);
    const float32Array = new Float32Array(int16Array.length);

    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }

    const audioBuffer = audioContext.createBuffer(1, float32Array.length, 24000);
    audioBuffer.getChannelData(0).set(float32Array);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const currentTime = audioContext.currentTime;
    if (nextStartTime < currentTime) {
        nextStartTime = currentTime;
    }

    source.start(nextStartTime);
    nextStartTime += audioBuffer.duration;
    audioEndTime = nextStartTime;
}

// 5. Server Event Handling
function handleServerEvent(msg) {
    const { type, payload } = msg;

    if (type === 'SESSION_READY') {
        updateFsmBadge(payload.fsm_state);
        appendAuditLog(`[Session Ready] Model: ${payload.model}, Project: ${payload.project_id}`);
    } else if (type === 'TRANSCRIPT_CHUNK') {
        const speaker = payload.speaker || 'Gemini';
        appendTranscript(speaker, payload.text);
    } else if (type === 'DYNAMIC_CONTEXT_INJECTION') {
        injectionCount++;
        injectionCountBadge.textContent = `${injectionCount} Injections`;
        updateFsmBadge(payload.fsm_state);
        appendContextCard(payload);
        appendAuditLog(`[DYNAMIC_CONTEXT_INJECTION] Event: ${payload.event_type}, State: ${payload.fsm_state}`);
    } else if (type === 'KILL_SWITCH_TRIGGERED') {
        appendAuditLog(`[KILL_SWITCH_TRIGGERED] Mode: ${payload.mode}, Reason: ${payload.reason}`);
    }
}

function updateFsmBadge(state) {
    fsmBadge.textContent = `FSM: ${state}`;
    if (state === 'BEREAVEMENT_HARDSHIP_CARE') {
        fsmBadge.style.borderColor = 'var(--accent-purple)';
        fsmBadge.style.color = 'var(--accent-purple)';
    } else if (state === 'INVESTMENT_DISCLAIMER_GATE') {
        fsmBadge.style.borderColor = 'var(--accent-amber)';
        fsmBadge.style.color = 'var(--accent-amber)';
    } else {
        fsmBadge.style.borderColor = 'var(--accent-cyan)';
        fsmBadge.style.color = 'var(--accent-cyan)';
    }
}

function appendTranscript(speaker, text) {
    const placeholder = transcriptBox.querySelector('.system-msg');
    if (placeholder) placeholder.remove();

    const div = document.createElement('div');
    div.className = 'transcript-chunk';
    div.innerHTML = `<strong>${speaker}:</strong> ${text}`;
    transcriptBox.appendChild(div);
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function appendContextCard(payload) {
    const placeholder = contextFeed.querySelector('.feed-placeholder');
    if (placeholder) placeholder.remove();

    const card = document.createElement('div');
    card.className = `injection-card ${payload.event_type}`;

    card.innerHTML = `
        <div class="card-title-bar">
            <span>[INJECTION TRIGGER]: ${payload.event_type}</span>
            <span>State: ${payload.fsm_state}</span>
        </div>
        <div class="card-body"><strong>Grounding Policy:</strong>\n${payload.grounding_data}</div>
        <div class="card-body"><strong>System Override:</strong>\n${payload.system_override}</div>
    `;

    contextFeed.prepend(card);
}

function appendAuditLog(text) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const timestamp = new Date().toISOString();
    entry.textContent = `[${timestamp}] ${text}`;
    auditLogBox.appendChild(entry);
    auditLogBox.scrollTop = auditLogBox.scrollHeight;
}

// 6. Simulation Button Actions
btnConnect.addEventListener('click', toggleSession);
btnMute.addEventListener('click', toggleMute);

btnSimBereavement.addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('Please click "Start Voice Session" first before running triggers.');
        return;
    }
    ws.send(JSON.stringify({
        action: 'simulate_disclosure',
        event_type: 'BEREAVEMENT',
        summary: 'Customer stated husband passed away, asking about mortgage payment pause.'
    }));
});

btnSimAdvice.addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('Please click "Start Voice Session" first before running triggers.');
        return;
    }
    ws.send(JSON.stringify({
        action: 'simulate_disclosure',
        event_type: 'UNAUTHORIZED_INVESTMENT_ADVICE',
        summary: 'Customer asked if they should invest $50k in TSLA stock.'
    }));
});

btnSimKillSwitch.addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('Please click "Start Voice Session" first before running triggers.');
        return;
    }
    ws.send(JSON.stringify({
        action: 'trigger_kill_switch',
        mode: 'SOFT_HANDOFF',
        reason: 'Manual kill switch trigger from testing UI panel'
    }));
});

// Text Speech Prompt Handler
const btnSendText = document.getElementById('btnSendText');
const textPromptInput = document.getElementById('textPromptInput');

function sendTextPrompt() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('Please click "Start Voice Session" first before sending speech prompts.');
        return;
    }
    const val = textPromptInput.value.trim();
    if (!val) return;

    appendTranscript('You', val);
    ws.send(JSON.stringify({
        action: 'send_text_prompt',
        text: val
    }));
    textPromptInput.value = '';
}

if (btnSendText) btnSendText.addEventListener('click', sendTextPrompt);
if (textPromptInput) {
    textPromptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendTextPrompt();
    });
}
