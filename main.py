import streamlit as st
import torch
import whisper
import numpy as np
import sounddevice as sd
import tempfile
import os
from datetime import datetime
import threading
import queue
import time

# Updated imports for proper threading context
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

class SpeechToTextAgent:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    @st.cache_resource
    def load_model(_self):
        """Load Whisper model with caching"""
        try:
            model = whisper.load_model("large-v3", device=_self.device)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        try:
            audio_data = sd.rec(int(duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1, 
                              dtype=np.float32)
            sd.wait()
            return audio_data.flatten()
        except Exception as e:
            st.error(f"Error recording audio: {e}")
            return None
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        if self.model is None:
            self.model = self.load_model()
        
        if self.model is None:
            return "Model not loaded"
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # Save audio data
                import soundfile as sf
                sf.write(tmp_file.name, audio_data, self.sample_rate)
                
                # Transcribe
                start_time = time.time()
                result = self.model.transcribe(tmp_file.name, 
                                             language="en",  # Change as needed
                                             fp16=True)  # Use fp16 for faster inference
                end_time = time.time()
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return {
                    "text": result["text"],
                    "processing_time": end_time - start_time,
                    "language": result.get("language", "unknown")
                }
        except Exception as e:
            return f"Error transcribing: {e}"
    
    def real_time_transcribe(self, ctx):
        """Real-time transcription with streaming - FIXED with proper context"""
        # Add Streamlit context to this thread
        add_script_run_ctx(ctx)
        
        chunk_duration = 3  # seconds
        
        try:
            while self.is_recording:
                audio_chunk = self.record_audio(chunk_duration)
                if audio_chunk is not None:
                    result = self.transcribe_audio(audio_chunk)
                    if isinstance(result, dict):
                        self.audio_queue.put(result)
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in real_time_transcribe: {e}")
            self.is_recording = False

def main():
    st.set_page_config(
        page_title="Speech-to-Text Agent",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Advanced Speech-to-Text Agent")
    st.markdown("**Powered by OpenAI Whisper Large-v3 on A40 GPU**")
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = SpeechToTextAgent()
    
    # GPU Status
    col1, col2, col3 = st.columns(3)
    with col1:
        device = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔴 CPU"
        st.metric("Device", device)
    
    with col2:
        if torch.cuda.is_available():
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            st.metric("GPU Memory", gpu_memory)
    
    with col3:
        model_status = "✅ Ready" if st.session_state.agent.model else "⏳ Loading"
        st.metric("Model Status", model_status)
    
    # Main interface
    st.markdown("---")
    
    # Mode selection
    mode = st.radio("Select Mode:", ["Single Recording", "Real-time Streaming"])
    
    if mode == "Single Recording":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            duration = st.slider("Recording Duration (seconds)", 1, 30, 5)
            
            if st.button("🎤 Start Recording", type="primary"):
                with st.spinner("Recording..."):
                    progress_bar = st.progress(0)
                    for i in range(duration):
                        time.sleep(1)
                        progress_bar.progress((i + 1) / duration)
                    
                    audio_data = st.session_state.agent.record_audio(duration)
                    
                if audio_data is not None:
                    with st.spinner("Transcribing..."):
                        result = st.session_state.agent.transcribe_audio(audio_data)
                    
                    if isinstance(result, dict):
                        st.success("✅ Transcription Complete!")
                        st.info(f"**Processing Time:** {result['processing_time']:.2f} seconds")
                        st.info(f"**Detected Language:** {result['language']}")
                        
                        # Display result
                        st.text_area("Transcription:", result["text"], height=150)
                        
                        # Save option
                        if st.button("💾 Save Transcription"):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"transcription_{timestamp}.txt"
                            with open(filename, "w") as f:
                                f.write(result["text"])
                            st.success(f"Saved as {filename}")
                    else:
                        st.error(result)
        
        with col2:
            st.markdown("### 📊 Performance Metrics")
            if torch.cuda.is_available():
                try:
                    gpu_util = f"{torch.cuda.utilization():.1f}%"
                    gpu_memory_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    st.metric("GPU Utilization", gpu_util)
                    st.metric("GPU Memory Used", gpu_memory_used)
                except Exception as e:
                    st.metric("GPU Utilization", "Not available")
                    st.metric("GPU Memory Used", "Not available")
    
    else:  # Real-time streaming
        st.markdown("### 🔴 Real-time Streaming Mode")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.get('streaming', False):
                if st.button("🔴 Start Streaming", type="primary"):
                    st.session_state.streaming = True
                    st.session_state.agent.is_recording = True
                    
                    # FIXED: Get current context and pass it to thread
                    ctx = get_script_run_ctx()
                    
                    # Start background thread with proper context
                    thread = threading.Thread(
                        target=st.session_state.agent.real_time_transcribe,
                        args=(ctx,),
                        daemon=True
                    )
                    thread.start()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Streaming", type="secondary"):
                    st.session_state.streaming = False
                    st.session_state.agent.is_recording = False
                    st.rerun()
        
        # Display streaming results
        if st.session_state.get('streaming', False):
            st.markdown("**🎙️ Listening...**")
            
            # Create placeholder for results
            results_container = st.container()
            
            # Process queue - FIXED: No direct st. calls from thread
            transcription_results = []
            while not st.session_state.agent.audio_queue.empty():
                try:
                    result = st.session_state.agent.audio_queue.get_nowait()
                    transcription_results.append(result)
                except queue.Empty:
                    break
            
            # Display results in main thread
            if transcription_results:
                with results_container:
                    for i, result in enumerate(transcription_results[-3:]):  # Show last 3 results
                        st.text_area(f"Transcription {i+1}:", result["text"], height=100)
                        st.caption(f"Processing time: {result['processing_time']:.2f}s")
            
            # Auto-refresh for streaming
            time.sleep(1)
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Tips for best results:**")
    st.markdown("- Speak clearly and at moderate pace")
    st.markdown("- Minimize background noise")
    st.markdown("- Ensure good microphone quality")
    st.markdown("- For technical terms, speak slowly")

if __name__ == "__main__":
    main()
