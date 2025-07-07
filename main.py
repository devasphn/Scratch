import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import whisper
from aiohttp import web, web_request, WSMsgType
from aiohttp.web_ws import WebSocketResponse
from silero_vad_lite import SileroVAD
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeechToTextEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        self.vad_model = None
        self.sample_rate = 16000
        self.audio_buffer = []
        self.is_recording = False
        
        logger.info(f"Initializing Speech-to-Text Engine on {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and Silero VAD models"""
        try:
            # Load Whisper model
            logger.info("Loading Whisper Large-v3 model...")
            self.whisper_model = whisper.load_model("large-v3", device=self.device)
            logger.info("✅ Whisper model loaded successfully")
            
            # Load Silero VAD model
            logger.info("Loading Silero VAD model...")
            self.vad_model = SileroVAD(self.sample_rate)
            logger.info("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def detect_speech(self, audio_chunk):
        """Use Silero VAD to detect speech in audio chunk"""
        try:
            # Convert audio to numpy array if needed
            if isinstance(audio_chunk, bytes):
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
            else:
                audio_array = np.array(audio_chunk, dtype=np.float32)
            
            # Ensure audio is the right length for VAD
            if len(audio_array) == 0:
                return 0.0
            
            # Get speech probability from Silero VAD
            speech_prob = self.vad_model.process(audio_array)
            
            logger.debug(f"Speech probability: {speech_prob:.3f}")
            return speech_prob
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            return 0.0
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        try:
            if len(audio_data) == 0:
                return {"error": "No audio data provided"}
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Convert to numpy array if needed
                if isinstance(audio_data, list):
                    audio_array = np.concatenate(audio_data)
                else:
                    audio_array = np.array(audio_data, dtype=np.float32)
                
                # Save audio file
                sf.write(tmp_file.name, audio_array, self.sample_rate)
                
                # Transcribe with Whisper
                start_time = time.time()
                result = self.whisper_model.transcribe(
                    tmp_file.name,
                    language="en",
                    fp16=True,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6
                )
                end_time = time.time()
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", "en"),
                    "processing_time": round(end_time - start_time, 2),
                    "audio_duration": len(audio_array) / self.sample_rate,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"error": str(e)}

# Global speech engine instance
speech_engine = SpeechToTextEngine()

async def websocket_handler(request):
    """Handle WebSocket connections for real-time audio processing"""
    ws = WebSocketResponse()
    await ws.prepare(request)
    
    logger.info("🔗 New WebSocket connection established")
    
    # Audio processing state
    audio_buffer = []
    speech_segments = []
    silence_counter = 0
    speech_threshold = 0.5  # Silero VAD threshold
    silence_threshold = 10  # Number of silent chunks before processing
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message_type = data.get('type')
                    
                    if message_type == 'audio_chunk':
                        # Receive audio chunk from browser
                        audio_chunk = np.array(data['audio'], dtype=np.float32)
                        
                        # Use Silero VAD to detect speech
                        speech_prob = speech_engine.detect_speech(audio_chunk)
                        
                        if speech_prob > speech_threshold:
                            # Speech detected - add to buffer
                            audio_buffer.append(audio_chunk)
                            silence_counter = 0
                            
                            # Send VAD feedback to client
                            await ws.send_text(json.dumps({
                                'type': 'vad_result',
                                'speech_detected': True,
                                'speech_probability': float(speech_prob),
                                'buffer_length': len(audio_buffer)
                            }))
                            
                        else:
                            # Silence detected
                            silence_counter += 1
                            
                            await ws.send_text(json.dumps({
                                'type': 'vad_result',
                                'speech_detected': False,
                                'speech_probability': float(speech_prob),
                                'silence_counter': silence_counter
                            }))
                            
                            # If we have speech in buffer and enough silence, process it
                            if len(audio_buffer) > 0 and silence_counter >= silence_threshold:
                                logger.info(f"🎤 Processing speech segment with {len(audio_buffer)} chunks")
                                
                                # Transcribe the accumulated speech
                                result = speech_engine.transcribe_audio(audio_buffer)
                                
                                if "error" not in result and result["text"]:
                                    logger.info(f"📝 Transcription: {result['text']}")
                                    
                                    # Send transcription result
                                    await ws.send_text(json.dumps({
                                        'type': 'transcription',
                                        'result': result
                                    }))
                                
                                # Reset buffer
                                audio_buffer = []
                                silence_counter = 0
                    
                    elif message_type == 'start_recording':
                        logger.info("🎙️ Started recording")
                        audio_buffer = []
                        silence_counter = 0
                        
                        await ws.send_text(json.dumps({
                            'type': 'status',
                            'message': 'Recording started'
                        }))
                    
                    elif message_type == 'stop_recording':
                        logger.info("⏹️ Stopped recording")
                        
                        # Process any remaining audio in buffer
                        if len(audio_buffer) > 0:
                            result = speech_engine.transcribe_audio(audio_buffer)
                            if "error" not in result and result["text"]:
                                await ws.send_text(json.dumps({
                                    'type': 'transcription',
                                    'result': result
                                }))
                        
                        audio_buffer = []
                        await ws.send_text(json.dumps({
                            'type': 'status',
                            'message': 'Recording stopped'
                        }))
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
                break
    
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    
    finally:
        logger.info("🔌 WebSocket connection closed")
    
    return ws

async def index_handler(request):
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 WebRTC Speech-to-Text with Silero VAD</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            min-width: 150px;
        }
        
        .record-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
        }
        
        .stop-btn {
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            color: white;
        }
        
        .record-btn:hover, .stop-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.2);
            font-size: 18px;
            font-weight: bold;
        }
        
        .vad-status {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .vad-item {
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 10px;
            margin: 5px;
            text-align: center;
        }
        
        .transcription {
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            min-height: 100px;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .transcription-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 4px solid #4ecdc4;
        }
        
        .timestamp {
            font-size: 12px;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .metric {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 14px;
            opacity: 0.8;
        }
        
        .speech-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
            transition: all 0.3s ease;
        }
        
        .speech-active {
            background: #4ecdc4;
            box-shadow: 0 0 20px #4ecdc4;
        }
        
        .speech-inactive {
            background: #ff6b6b;
        }
        
        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            button {
                width: 100%;
                max-width: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 WebRTC Speech-to-Text with Silero VAD</h1>
        
        <div class="controls">
            <button id="startBtn" class="record-btn">🎙️ Start Recording</button>
            <button id="stopBtn" class="stop-btn" disabled>⏹️ Stop Recording</button>
        </div>
        
        <div id="status" class="status">Ready to record</div>
        
        <div class="vad-status">
            <div class="vad-item">
                <div>Speech Detection</div>
                <div><span id="speechIndicator" class="speech-indicator speech-inactive"></span><span id="speechStatus">No Speech</span></div>
            </div>
            <div class="vad-item">
                <div>Speech Probability</div>
                <div id="speechProb">0.00</div>
            </div>
            <div class="vad-item">
                <div>Buffer Length</div>
                <div id="bufferLength">0</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div id="totalTranscriptions" class="metric-value">0</div>
                <div class="metric-label">Total Transcriptions</div>
            </div>
            <div class="metric">
                <div id="avgProcessingTime" class="metric-value">0.0s</div>
                <div class="metric-label">Avg Processing Time</div>
            </div>
            <div class="metric">
                <div id="connectionStatus" class="metric-value">Disconnected</div>
                <div class="metric-label">Connection Status</div>
            </div>
        </div>
        
        <div class="transcription">
            <h3>📝 Transcriptions</h3>
            <div id="transcriptions">
                <p style="text-align: center; opacity: 0.7;">Transcriptions will appear here...</p>
            </div>
        </div>
    </div>

    <script>
        class SpeechToTextClient {
            constructor() {
                this.ws = null;
                this.audioContext = null;
                this.mediaStream = null;
                this.processor = null;
                this.isRecording = false;
                this.transcriptionCount = 0;
                this.totalProcessingTime = 0;
                
                this.initializeElements();
                this.connectWebSocket();
            }
            
            initializeElements() {
                this.startBtn = document.getElementById('startBtn');
                this.stopBtn = document.getElementById('stopBtn');
                this.status = document.getElementById('status');
                this.transcriptions = document.getElementById('transcriptions');
                this.speechIndicator = document.getElementById('speechIndicator');
                this.speechStatus = document.getElementById('speechStatus');
                this.speechProb = document.getElementById('speechProb');
                this.bufferLength = document.getElementById('bufferLength');
                this.totalTranscriptionsEl = document.getElementById('totalTranscriptions');
                this.avgProcessingTimeEl = document.getElementById('avgProcessingTime');
                this.connectionStatusEl = document.getElementById('connectionStatus');
                
                this.startBtn.addEventListener('click', () => this.startRecording());
                this.stopBtn.addEventListener('click', () => this.stopRecording());
            }
            
            connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.connectionStatusEl.textContent = 'Connected';
                    this.status.textContent = 'Connected - Ready to record';
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.connectionStatusEl.textContent = 'Disconnected';
                    this.status.textContent = 'Disconnected - Refresh page to reconnect';
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.connectionStatusEl.textContent = 'Error';
                };
            }
            
            handleWebSocketMessage(data) {
                switch(data.type) {
                    case 'vad_result':
                        this.updateVADStatus(data);
                        break;
                    case 'transcription':
                        this.addTranscription(data.result);
                        break;
                    case 'status':
                        this.status.textContent = data.message;
                        break;
                }
            }
            
            updateVADStatus(data) {
                if (data.speech_detected) {
                    this.speechIndicator.className = 'speech-indicator speech-active';
                    this.speechStatus.textContent = 'Speech Detected';
                } else {
                    this.speechIndicator.className = 'speech-indicator speech-inactive';
                    this.speechStatus.textContent = 'No Speech';
                }
                
                this.speechProb.textContent = data.speech_probability.toFixed(3);
                this.bufferLength.textContent = data.buffer_length || data.silence_counter || 0;
            }
            
            addTranscription(result) {
                if (!result.text.trim()) return;
                
                this.transcriptionCount++;
                this.totalProcessingTime += result.processing_time;
                
                // Update metrics
                this.totalTranscriptionsEl.textContent = this.transcriptionCount;
                this.avgProcessingTimeEl.textContent = 
                    (this.totalProcessingTime / this.transcriptionCount).toFixed(2) + 's';
                
                // Add transcription to display
                const transcriptionDiv = document.createElement('div');
                transcriptionDiv.className = 'transcription-item';
                
                const timestamp = new Date(result.timestamp).toLocaleTimeString();
                
                transcriptionDiv.innerHTML = `
                    <div class="timestamp">${timestamp} | Processing: ${result.processing_time}s | Duration: ${result.audio_duration.toFixed(2)}s</div>
                    <div>${result.text}</div>
                `;
                
                if (this.transcriptions.children.length === 1 && 
                    this.transcriptions.children[0].tagName === 'P') {
                    this.transcriptions.innerHTML = '';
                }
                
                this.transcriptions.insertBefore(transcriptionDiv, this.transcriptions.firstChild);
                
                // Keep only last 10 transcriptions
                while (this.transcriptions.children.length > 10) {
                    this.transcriptions.removeChild(this.transcriptions.lastChild);
                }
            }
            
            async startRecording() {
                try {
                    this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        } 
                    });
                    
                    this.audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
                    
                    // Create ScriptProcessorNode for audio processing
                    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
                    
                    this.processor.onaudioprocess = (event) => {
                        if (this.isRecording && this.ws.readyState === WebSocket.OPEN) {
                            const audioData = event.inputBuffer.getChannelData(0);
                            
                            this.ws.send(JSON.stringify({
                                type: 'audio_chunk',
                                audio: Array.from(audioData)
                            }));
                        }
                    };
                    
                    source.connect(this.processor);
                    this.processor.connect(this.audioContext.destination);
                    
                    this.isRecording = true;
                    this.startBtn.disabled = true;
                    this.stopBtn.disabled = false;
                    
                    // Send start recording message
                    this.ws.send(JSON.stringify({ type: 'start_recording' }));
                    
                    this.status.textContent = '🎙️ Recording... Speak now!';
                    
                } catch (error) {
                    console.error('Error starting recording:', error);
                    this.status.textContent = 'Error: Could not access microphone';
                }
            }
            
            stopRecording() {
                this.isRecording = false;
                
                if (this.processor) {
                    this.processor.disconnect();
                    this.processor = null;
                }
                
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
                
                if (this.mediaStream) {
                    this.mediaStream.getTracks().forEach(track => track.stop());
                    this.mediaStream = null;
                }
                
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                
                // Send stop recording message
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'stop_recording' }));
                }
                
                this.status.textContent = 'Recording stopped';
                this.speechIndicator.className = 'speech-indicator speech-inactive';
                this.speechStatus.textContent = 'No Speech';
            }
        }
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', () => {
            new SpeechToTextClient();
        });
    </script>
</body>
</html>
    """
    return web.Response(text=html_content, content_type='text/html')

async def health_handler(request):
    """Health check endpoint"""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return web.json_response({
        "status": "healthy",
        "whisper_model": "large-v3",
        "vad_model": "silero",
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat()
    })

def create_app():
    """Create and configure the web application"""
    app = web.Application()
    
    # Add routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/health', health_handler)
    
    return app

if __name__ == '__main__':
    # Create application
    app = create_app()
    
    # Run server
    logger.info("🚀 Starting WebRTC Speech-to-Text Server")
    logger.info(f"🎯 Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    logger.info("🌐 Server will be available at http://YOUR_RUNPOD_IP:8080")
    
    web.run_app(app, host='0.0.0.0', port=8080)
