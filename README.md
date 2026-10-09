# Dynamic Context Injection for Real-Time Voice AI in Financial Services

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Gen AI SDK](https://img.shields.io/badge/SDK-google--genai-green.svg)](https://github.com/googleapis/python-genai)
[![Gemini Live API](https://img.shields.io/badge/Model-gemini--live--2.5--flash--native--audio-orange.svg)](https://cloud.google.com/vertex-ai)

> **A production-ready FastAPI WebSocket application built on the official `google-genai` SDK and Vertex AI (`gemini-live-2.5-flash-native-audio`) that streams real-time bidirectional voice audio while executing sub-second native function calling, dynamic prompt context injection, financial compliance guardrails, and SEC Rule 17a-4 audit logging.**

---

## 📖 Deep-Dive Article

This repository contains the complete reference implementation and code snippets for the technical article:

👉 **Read the full deep-dive article on Medium:** **[Dynamic Context Injection for Real-Time Voice AI in Financial Services](<MEDIUM_ARTICLE_URL>)**

---

## 🎯 Purpose & Problem Statement

In real-time multimodal voice streams, traditional multi-stage interception pipelines (STT → Moderation LLM → Core LLM → TTS) introduce **1.5 to 3 seconds of latency**, destroying natural conversational cadence. 

When customers experience acute trauma (bereavement, severe illness) or shift to restricted financial topics (unauthorized stock recommendations, debt default advice), voice AI systems must dynamically inject **fresh prompt context, legal disclaimers, and trauma-informed grounding data** mid-session without introducing conversational lag or dropping the WebSocket connection.

This project implements **Native Gemini Function-Calling Context Injection**, a streamlined architecture that leverages Gemini's native audio prosody and intent understanding over a single bidirectional WebSocket stream (<800ms time-to-first-audio packet).

---

## 🚀 Key Features

* **Single-Stream Multimodal WebSockets:** Communicates directly with Vertex AI (`gemini-live-2.5-flash-native-audio`) sending raw 16kHz PCM audio input and receiving 24kHz audio + real-time transcriptions.
* **Sub-Second Dynamic Context Injection (<125ms Overhead):** Intercepts Gemini tool calls mid-session, retrieves RAG policy grounding data, and pushes system prompt overrides dynamically via `session.send_client_content(role="system")` without session teardown.
* **Trauma-Informed & Financial Compliance FSM:** Supports automated state routing for bereavement disclosures, medical hardship, FINRA Rule 2210 investment disclaimers, and CFPB vulnerable customer protections.
* **Immutable SEC Rule 17a-4 & CFPB Audit Ledger:** Writes every tool invocation, injected policy clause, prompt override, and timestamp to an append-only JSONL ledger (`audit_ledger.jsonl`).
* **Multi-Tier Kill Switch Circuit Breaker:** Supports soft handoffs (warm transfers to human specialist care) and hard socket terminations for emergency safety overrides.
* **Interactive Browser UI & Simulation Suite:** Includes a full-featured SSL web frontend (`https://localhost:8000`) with audio visualizer, real-time transcript panel, and UI trigger controls.

---

## 🏗️ System Architecture

```
  [ Browser Client / Microphone ]
                 |
                 v  (16kHz PCM Audio Stream over WSS)
  +-------------------------------------------------------------+
  |              FastAPI WebSocket Server (server.py)           |
  |               Google Gen AI SDK (google-genai)              |
  +------------------------------+------------------------------+
                                 |
                                 v  (BidiGenerateContent)
  +-------------------------------------------------------------+
  |               Gemini Live API WebSocket Session             |
  |             (gemini-live-2.5-flash-native-audio)             |
  |                                                             |
  |  1. Hears vocal distress / topic drift in raw PCM audio     |
  |  2. Emits FunctionCall: report_customer_disclosure(...)     |
  +------------------------------+------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |                   Client Function Handler                   |
  |  1. Fetches Hardship RAG / Regulatory Disclaimers          |
  |  2. Sends LiveClientToolResponse with Grounding Payload      |
  |  3. Sends session.send_client_content(role="system")        |
  |     to update system instructions for future turns          |
  +------------------------------+------------------------------+
                                 |
                                 v
              [ Immutable Audit Log (FINRA/SEC/CFPB) ]
```

---

## 📁 Repository Structure

```
.
├── README.md                           # Project documentation
├── dynamic_context_injection_article.md# Full technical article source
├── server.py                           # FastAPI SSL WebSocket server & Live API bridge
├── voice_compliance_agent.py           # Standalone orchestrator & SEC 17a-4 audit ledger
├── static/                             # Browser UI (HTML5, Web Audio API, Tailwind)
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── cert.pem / key.pem                  # Self-signed TLS certificates for WSS localhost
├── requirements.txt                    # Python dependencies
├── audit_ledger.jsonl                  # Append-only compliance audit trail
└── test_*.py                           # Automated test & simulation suite
```

---

## 🛠️ Quickstart Guide

### Prerequisites
* Python 3.10+
* Google Cloud Project with **Vertex AI API** enabled (`aiplatform.googleapis.com`)
* `gcloud` CLI installed and authenticated

### 1. Installation
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/your-username/dynamic-context-injection.git
cd dynamic-context-injection

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Google Cloud Authentication
Authenticate with your Google Cloud credentials and set target project variables:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 3. Run the Backend Server
Start the HTTPS / Secure WebSocket (WSS) server:

```bash
python3 server.py
```

The server will launch at `https://localhost:8000`.

### 4. Access the Browser Interface
1. Open your browser and navigate to `https://localhost:8000`.
2. Accept the self-signed SSL certificate warning if prompted.
3. Click **Connect Agent** to establish the live Gemini WebSocket session.
4. Speak into your microphone (e.g., *"My spouse passed away last week, how do I pause my mortgage?"*) to observe real-time dynamic context injection and audit logging!

---

## 🧪 Simulation & Automated Testing

You can run automated simulation tests without opening a browser or microphone:

```bash
# Test standalone tool call & audit ledger execution
python3 test_simulation.py

# Test full multi-turn WebSocket pipeline
python3 test_ws_pipeline.py
```

---

## 📜 License

This repository is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
Article documentation is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
