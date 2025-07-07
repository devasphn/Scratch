import streamlit as st
import torch
import whisper
import numpy as np
import sounddevice as sd
import tempfile
import os
from datetime import datetime
import queue
import time

class SpeechToTextAgent:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                import soundfile as sf
                sf.write(tmp_file.name, audio_data, self.sample_rate)
                
                start_time = time.time()
                result = self.model.transcribe(tmp_file.name, 
                                             language="en",
                                             fp16=True)
                end_time = time.time()
                
                os.unlink(tmp_file.name)
                
                return {
                    "text": result["text"],
                    "processing_time": end_time - start_time,
                    "language": result.get("language", "unknown")
                }
        except Exception as e:
            return f"Error transcribing: {e}"

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
    
    # Initialize session state for streaming
    if 'streaming_results' not in st.session_state:
        st.session_state.streaming_results = []
    
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
    mode = st.radio("Select Mode:", ["Single Recording", "Continuous Recording"])
    
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
                except Exception:
                    st.metric("GPU Utilization", "Not available")
                    st.metric("GPU Memory Used", "Not available")
    
    else:  # Continuous Recording - NO THREADING
        st.markdown("### 🔴 Continuous Recording Mode")
        st.markdown("*Click 'Record Chunk' repeatedly for continuous transcription*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_duration = st.slider("Chunk Duration (seconds)", 1, 10, 3)
            
            if st.button("🎤 Record Chunk", type="primary"):
                with st.spinner(f"Recording {chunk_duration} seconds..."):
                    audio_data = st.session_state.agent.record_audio(chunk_duration)
                
                if audio_data is not None:
                    with st.spinner("Transcribing..."):
                        result = st.session_state.agent.transcribe_audio(audio_data)
                    
                    if isinstance(result, dict):
                        # Add to streaming results
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        result["timestamp"] = timestamp
                        st.session_state.streaming_results.append(result)
                        
                        # Keep only last 10 results
                        if len(st.session_state.streaming_results) > 10:
                            st.session_state.streaming_results = st.session_state.streaming_results[-10:]
                        
                        st.success(f"✅ Chunk transcribed in {result['processing_time']:.2f}s")
                        st.rerun()
            
            if st.button("🗑️ Clear Results"):
                st.session_state.streaming_results = []
                st.rerun()
        
        with col2:
            st.markdown("### 📝 Transcription History")
            
            if st.session_state.streaming_results:
                for i, result in enumerate(reversed(st.session_state.streaming_results)):
                    with st.expander(f"🎤 {result['timestamp']} - {result['processing_time']:.2f}s"):
                        st.text_area("Text:", result["text"], height=100, key=f"result_{i}")
            else:
                st.info("No transcriptions yet. Click 'Record Chunk' to start.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Tips for best results:**")
    st.markdown("- Speak clearly and at moderate pace")
    st.markdown("- Minimize background noise")
    st.markdown("- Ensure good microphone quality")
    st.markdown("- For technical terms, speak slowly")

if __name__ == "__main__":
    main()
