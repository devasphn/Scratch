import streamlit as st
import whisper
import torch
import tempfile
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install the audio recorder component
# pip install streamlit-audiorec

from st_audiorec import st_audiorec

class SpeechToTextAgent:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def load_model(_self):
        """Load Whisper model with caching"""
        try:
            logger.info("Loading Whisper large-v3 model...")
            model = whisper.load_model("large-v3", device=_self.device)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"Error loading model: {e}")
            return None
    
    def transcribe_audio(self, audio_bytes):
        """Transcribe audio using Whisper"""
        if self.model is None:
            logger.info("Loading model for transcription...")
            self.model = self.load_model()
        
        if self.model is None:
            logger.error("Model not loaded")
            return "Model not loaded"
        
        try:
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                logger.info(f"Created temporary file: {tmp_file.name}")
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                
                # Transcribe
                logger.info("Starting transcription...")
                start_time = time.time()
                result = self.model.transcribe(tmp_file.name, 
                                             language="en",
                                             fp16=True)
                end_time = time.time()
                
                # Clean up
                os.unlink(tmp_file.name)
                logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
                
                return {
                    "text": result["text"],
                    "processing_time": end_time - start_time,
                    "language": result.get("language", "unknown")
                }
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error transcribing: {e}"

def main():
    st.set_page_config(
        page_title="Speech-to-Text Agent",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Advanced Speech-to-Text Agent")
    st.markdown("**Powered by OpenAI Whisper Large-v3 on A40 GPU**")
    st.markdown("**Browser-Based Microphone Recording**")
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = SpeechToTextAgent()
        logger.info("Initialized SpeechToTextAgent")
    
    # Initialize transcription history
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []
    
    # GPU Status
    col1, col2, col3 = st.columns(3)
    with col1:
        device = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔴 CPU"
        st.metric("Device", device)
        logger.info(f"Using device: {device}")
    
    with col2:
        if torch.cuda.is_available():
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            st.metric("GPU Memory", gpu_memory)
    
    with col3:
        model_status = "✅ Ready" if st.session_state.agent.model else "⏳ Loading"
        st.metric("Model Status", model_status)
    
    # Main interface
    st.markdown("---")
    
    # Browser-based audio recording
    st.markdown("### 🎤 Record Audio")
    st.markdown("**Click the record button below to start recording from your microphone**")
    
    # Audio recorder component
    wav_audio_data = st_audiorec()
    
    if wav_audio_data is not None:
        logger.info("Audio data received from browser")
        st.success("✅ Audio recorded successfully!")
        
        # Display audio player
        st.audio(wav_audio_data, format='audio/wav')
        
        # Transcription section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Transcribe Audio", type="primary"):
                logger.info("Starting transcription process...")
                with st.spinner("Transcribing audio..."):
                    result = st.session_state.agent.transcribe_audio(wav_audio_data)
                
                if isinstance(result, dict):
                    st.success("✅ Transcription Complete!")
                    
                    # Display results
                    st.info(f"**Processing Time:** {result['processing_time']:.2f} seconds")
                    st.info(f"**Detected Language:** {result['language']}")
                    
                    # Main transcription display
                    st.text_area("Transcription:", result["text"], height=150, key="main_transcription")
                    
                    # Add to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.transcription_history.append({
                        "timestamp": timestamp,
                        "text": result["text"],
                        "processing_time": result["processing_time"],
                        "language": result["language"]
                    })
                    
                    logger.info(f"Transcription added to history: {result['text'][:50]}...")
                    
                else:
                    st.error(result)
                    logger.error(f"Transcription failed: {result}")
        
        with col2:
            if st.button("💾 Save Transcription"):
                if isinstance(result, dict):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"transcription_{timestamp}.txt"
                    with open(filename, "w") as f:
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Processing Time: {result['processing_time']:.2f}s\n")
                        f.write(f"Language: {result['language']}\n")
                        f.write(f"Text: {result['text']}\n")
                    st.success(f"✅ Saved as {filename}")
                    logger.info(f"Transcription saved to {filename}")
                else:
                    st.warning("No transcription to save")
    
    # Transcription History
    if st.session_state.transcription_history:
        st.markdown("---")
        st.markdown("### 📝 Transcription History")
        
        for i, entry in enumerate(reversed(st.session_state.transcription_history[-5:])):
            with st.expander(f"🎤 {entry['timestamp']} - {entry['processing_time']:.2f}s"):
                st.text_area("Text:", entry["text"], height=100, key=f"history_{i}")
                st.caption(f"Language: {entry['language']}")
    
    # Performance Metrics
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if torch.cuda.is_available():
            try:
                gpu_util = f"{torch.cuda.utilization():.1f}%"
                st.metric("GPU Utilization", gpu_util)
            except Exception as e:
                st.metric("GPU Utilization", "Not available")
                logger.warning(f"GPU utilization not available: {e}")
    
    with col2:
        if torch.cuda.is_available():
            try:
                gpu_memory_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
                st.metric("GPU Memory Used", gpu_memory_used)
            except Exception as e:
                st.metric("GPU Memory Used", "Not available")
    
    with col3:
        st.metric("Total Transcriptions", len(st.session_state.transcription_history))
    
    # Clear history option
    if st.session_state.transcription_history:
        if st.button("🗑️ Clear History"):
            st.session_state.transcription_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Instructions
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
    1. **Click the record button** 🔴 to start recording
    2. **Speak clearly** into your microphone
    3. **Click stop** ⏹️ when finished
    4. **Click 'Transcribe Audio'** to convert speech to text
    5. **View results** and save if needed
    """)
    
    st.markdown("### 🔧 Troubleshooting")
    st.markdown("""
    - **No microphone prompt?** Make sure you're using **HTTPS** or add your domain to browser exceptions
    - **Recording not working?** Check browser microphone permissions
    - **Poor quality?** Speak clearly and minimize background noise
    """)

if __name__ == "__main__":
    import time
    main()
