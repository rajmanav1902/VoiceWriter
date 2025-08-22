import streamlit as st
import whisper
import tempfile
import os
from datetime import datetime, timedelta
import pandas as pd
import io
import json
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🎤 Advanced Speech Transcriber",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .transcription-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .timestamp {
        color: #007bff;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .speaker-tag {
        background-color: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .stats-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None

def load_whisper_model(model_size="base"):
    """Load Whisper model with caching"""
    try:
        if not st.session_state.model_loaded or st.session_state.whisper_model is None:
            with st.spinner(f"Loading Whisper {model_size} model... This may take a moment."):
                # Use CPU for free deployment compatibility
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st.session_state.whisper_model = whisper.load_model(model_size, device=device)
                st.session_state.model_loaded = True
                st.success(f"✅ Whisper {model_size} model loaded successfully on {device.upper()}")
        return st.session_state.whisper_model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return None

def format_timestamp(seconds):
    """Convert seconds to MM:SS.mmm format"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((seconds - total_seconds) * 1000)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def transcribe_audio(audio_file, model, language="auto", task="transcribe"):
    """Transcribe audio file using Whisper"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔊 Loading audio file...")
        progress_bar.progress(10)
        
        # Transcribe with timestamps
        status_text.text("🔄 Transcribing audio... This may take a few minutes.")
        progress_bar.progress(30)
        
        # Whisper transcription options
        options = {
            "task": task,  # "transcribe" or "translate"
            "language": None if language == "auto" else language,
            "verbose": False,
            "word_timestamps": True  # Enable word-level timestamps
        }
        
        result = model.transcribe(tmp_file_path, **options)
        
        progress_bar.progress(90)
        status_text.text("✨ Processing results...")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        progress_bar.progress(100)
        status_text.text("✅ Transcription completed!")
        
        return result
        
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

def create_detailed_transcript(result):
    """Create detailed transcript with word-level timestamps"""
    detailed_transcript = []
    
    for segment in result['segments']:
        segment_data = {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip(),
            'words': []
        }
        
        # Add word-level data if available
        if 'words' in segment:
            for word in segment['words']:
                word_data = {
                    'start': word.get('start', segment['start']),
                    'end': word.get('end', segment['end']),
                    'word': word.get('word', '').strip(),
                    'probability': word.get('probability', 0.0)
                }
                segment_data['words'].append(word_data)
        
        detailed_transcript.append(segment_data)
    
    return detailed_transcript

def display_transcript(detailed_transcript):
    """Display formatted transcript"""
    st.markdown("### 📝 Transcription Results")
    
    # Create scrollable transcript container
    transcript_html = '<div style="max-height: 400px; overflow-y: auto; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; border-left: 4px solid #007bff;">'
    
    for i, segment in enumerate(detailed_transcript):
        timestamp = format_timestamp(segment['start'])
        
        transcript_html += f'''
        <div style="margin-bottom: 1rem; padding: 0.5rem; background-color: white; border-radius: 0.3rem;">
            <div class="timestamp">[{timestamp}]</div>
            <div style="margin-top: 0.5rem; line-height: 1.6;">{segment['text']}</div>
        </div>
        '''
    
    transcript_html += '</div>'
    st.markdown(transcript_html, unsafe_allow_html=True)

def generate_export_files(result, detailed_transcript, filename):
    """Generate different export formats"""
    exports = {}
    base_name = Path(filename).stem
    
    # 1. Plain Text with timestamps
    txt_content = []
    for segment in detailed_transcript:
        timestamp = format_timestamp(segment['start'])
        txt_content.append(f"[{timestamp}] {segment['text']}")
    exports['txt'] = '\n\n'.join(txt_content)
    
    # 2. SRT Format (for subtitles)
    srt_content = []
    for i, segment in enumerate(detailed_transcript, 1):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n")
    exports['srt'] = '\n'.join(srt_content)
    
    # 3. JSON Format (detailed)
    json_data = {
        'metadata': {
            'filename': filename,
            'language': result.get('language', 'unknown'),
            'duration': max([seg['end'] for seg in detailed_transcript]) if detailed_transcript else 0,
            'total_segments': len(detailed_transcript)
        },
        'transcript': detailed_transcript
    }
    exports['json'] = json.dumps(json_data, indent=2, ensure_ascii=False)
    
    # 4. CSV Format
    csv_data = []
    for segment in detailed_transcript:
        csv_data.append({
            'Start Time': format_timestamp(segment['start']),
            'End Time': format_timestamp(segment['end']),
            'Duration': segment['end'] - segment['start'],
            'Text': segment['text'],
            'Word Count': len(segment['text'].split())
        })
    df = pd.DataFrame(csv_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    exports['csv'] = csv_buffer.getvalue()
    
    return exports

def format_srt_time(seconds):
    """Format time for SRT files (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((seconds - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    # Header
    st.title("🎤 Advanced Speech Transcriber")
    st.markdown("### Fast, accurate transcription with precise timestamps powered by OpenAI Whisper")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_size = st.selectbox(
            "Select Whisper Model",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower. 'base' is recommended for most use cases."
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            options=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "hi", "ar"],
            index=0,
            help="Select 'auto' for automatic language detection"
        )
        
        # Task selection
        task = st.selectbox(
            "Task",
            options=["transcribe", "translate"],
            index=0,
            help="Transcribe: Convert speech to text in original language\nTranslate: Convert speech to English text"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Supported formats**: MP3, WAV, M4A, FLAC, OGG
        - **Max file size**: 200MB
        - **Best quality**: Clear audio, minimal background noise
        - **Processing time**: ~1-3 minutes for 1 hour audio
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma'],
            help="Upload your audio file (max 200MB)"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.read()) / (1024 * 1024)  # MB
            uploaded_file.seek(0)  # Reset file pointer
            
            st.info(f"📁 **File**: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Audio player
            st.audio(uploaded_file, format='audio/' + uploaded_file.name.split('.')[-1])
            
            # Load model and transcribe
            if st.button("🚀 Start Transcription", type="primary"):
                if file_size > 200:
                    st.error("❌ File too large! Please use a file smaller than 200MB.")
                else:
                    # Load model
                    model = load_whisper_model(model_size)
                    
                    if model:
                        # Transcribe
                        result = transcribe_audio(uploaded_file, model, language, task)
                        
                        if result:
                            # Process results
                            detailed_transcript = create_detailed_transcript(result)
                            st.session_state.transcription = {
                                'result': result,
                                'detailed': detailed_transcript,
                                'filename': uploaded_file.name
                            }
                            st.rerun()
    
    with col2:
        # Statistics and info
        st.markdown("### 📊 Statistics")
        
        if st.session_state.transcription:
            result = st.session_state.transcription['result']
            detailed = st.session_state.transcription['detailed']
            
            # Calculate stats
            total_duration = max([seg['end'] for seg in detailed]) if detailed else 0
            total_words = sum(len(seg['text'].split()) for seg in detailed)
            total_segments = len(detailed)
            avg_words_per_minute = (total_words / (total_duration / 60)) if total_duration > 0 else 0
            
            # Display stats in boxes
            stats_html = f"""
            <div class="stats-container">
                <h4>📈 Transcription Stats</h4>
                <p><strong>Duration:</strong> {format_timestamp(total_duration)}</p>
                <p><strong>Total Words:</strong> {total_words:,}</p>
                <p><strong>Segments:</strong> {total_segments}</p>
                <p><strong>Words/Min:</strong> {avg_words_per_minute:.1f}</p>
                <p><strong>Language:</strong> {result.get('language', 'Unknown').upper()}</p>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
        else:
            st.info("Upload an audio file and start transcription to see statistics.")
    
    # Display transcription results
    if st.session_state.transcription:
        display_transcript(st.session_state.transcription['detailed'])
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        exports = generate_export_files(
            st.session_state.transcription['result'],
            st.session_state.transcription['detailed'],
            st.session_state.transcription['filename']
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📄 Download TXT",
                data=exports['txt'],
                file_name=f"{Path(st.session_state.transcription['filename']).stem}_transcript.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="🎬 Download SRT",
                data=exports['srt'],
                file_name=f"{Path(st.session_state.transcription['filename']).stem}_subtitles.srt",
                mime="text/plain"
            )
        
        with col3:
            st.download_button(
                label="📋 Download JSON",
                data=exports['json'],
                file_name=f"{Path(st.session_state.transcription['filename']).stem}_data.json",
                mime="application/json"
            )
        
        with col4:
            st.download_button(
                label="📊 Download CSV",
                data=exports['csv'],
                file_name=f"{Path(st.session_state.transcription['filename']).stem}_segments.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 <strong>Powered by OpenAI Whisper</strong> | Built with Streamlit | 
        <a href='https://github.com/openai/whisper' target='_blank'>Learn more about Whisper</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
