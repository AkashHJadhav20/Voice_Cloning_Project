# gradio_ui_compatible.py
import gradio as gr
import torch
import os
import sys
import time
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from TTS.api import TTS
    from utils.voice_manager import VoiceManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Initialize components
print("Initializing Voice Cloning System...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load TTS model
print("Loading XTTS-v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("✓ Model loaded successfully!")

voice_manager = VoiceManager()
output_dir = Path("generated_audio")
output_dir.mkdir(exist_ok=True)

# Available speakers
SPEAKERS = ['AB', 'AP', 'ME', 'DF']

def get_voice_status():
    """Get status of all voices"""
    status = {}
    for speaker in SPEAKERS:
        path = voice_manager.get_voice_path(speaker)
        if path and os.path.exists(path):
            # Get audio info
            import soundfile as sf
            audio, sr = sf.read(path)
            duration = len(audio) / sr
            status[speaker] = {
                'available': True,
                'path': path,
                'duration': f"{duration:.1f}s",
                'sample_rate': f"{sr}Hz"
            }
        else:
            status[speaker] = {
                'available': False,
                'path': None,
                'duration': 'N/A',
                'sample_rate': 'N/A'
            }
    return status

def generate_speech(voice, text, progress=gr.Progress()):
    """Generate speech from text"""
    if not voice:
        return None, "❌ Please select a voice", None
    
    if not text or text.strip() == "":
        return None, "❌ Please enter some text", None
    
    # Get voice path
    voice_path = voice_manager.get_voice_path(voice)
    if not voice_path or not os.path.exists(voice_path):
        return None, f"❌ Voice {voice} not found. Please register it first.", None
    
    try:
        progress(0.2, desc="Loading model...")
        
        # Clear GPU cache if needed
        if device == "cuda":
            torch.cuda.empty_cache()
        
        progress(0.4, desc="Generating speech...")
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{voice}_{timestamp}.wav"
        output_path = output_dir / filename
        
        # Generate speech
        start_time = time.time()
        
        tts.tts_to_file(
            text=text,
            speaker_wav=voice_path,
            language="en",
            file_path=str(output_path)
        )
        
        generation_time = time.time() - start_time
        
        progress(1.0, desc="Done!")
        
        # Get audio duration
        import soundfile as sf
        audio, sr = sf.read(output_path)
        audio_duration = len(audio) / sr
        
        status = f"✅ Generated in {generation_time:.1f}s | Duration: {audio_duration:.1f}s | Voice: {voice}"
        
        return str(output_path), status, filename
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", None

def update_voice_info(voice):
    """Update voice information display"""
    status = get_voice_status()
    if voice and voice in status and status[voice]['available']:
        return f"""
### Voice: {voice}
- **Status:** ✅ Available
- **Duration:** {status[voice]['duration']}
- **Sample Rate:** {status[voice]['sample_rate']}
        """
    elif voice:
        return f"""
### Voice: {voice}
- **Status:** ❌ Not Available
- Please register this voice first using `register_voices.py`
        """
    else:
        return "### No voice selected"

def refresh_voices():
    """Refresh the voice dropdown"""
    status = get_voice_status()
    available_voices = [s for s in SPEAKERS if status[s]['available']]
    return gr.Dropdown.update(choices=available_voices, value=available_voices[0] if available_voices else None)

def load_sample_text():
    """Load sample text"""
    return "Hello, this is a sample of my voice cloning project. The quality sounds amazing and very similar to the original speaker."

# Get initial voice status
initial_status = get_voice_status()
available_voices = [s for s in SPEAKERS if initial_status[s]['available']]
default_voice = available_voices[0] if available_voices else None

# Custom CSS to prevent IDM popups (simplified)
custom_css = """
/* Hide default download button */
.download-symbol {
    display: none !important;
}
/* Style for our custom download section */
.custom-download {
    margin-top: 10px;
    padding: 10px;
    background: #f0f0f0;
    border-radius: 5px;
}
"""

# Create Gradio interface
with gr.Blocks(title="Voice Cloning Studio", css=custom_css) as demo:
    gr.Markdown("""
    # 🎤 Voice Cloning Studio
    ### Text-to-Speech with Your Own Voices (AB, AP, ME, DF)
    """)
    
    with gr.Row():
        # Device info
        device_info = f"**Device:** `{device}`"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            device_info += f" | **GPU:** `{gpu_name}`"
        gr.Markdown(device_info)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Voice Selection Panel
            gr.Markdown("### 🎤 Select Voice")
            voice_dropdown = gr.Dropdown(
                choices=available_voices,
                value=default_voice,
                label="Voice"
            )
            
            refresh_btn = gr.Button("🔄 Refresh Voices")
            
            # Voice Info Panel
            gr.Markdown("### 📊 Voice Information")
            voice_info = gr.Markdown(
                update_voice_info(default_voice) if default_voice else "### No voices available"
            )
            
            # Voice Status Table
            gr.Markdown("### 📋 All Voices Status")
            status_data = []
            for speaker in SPEAKERS:
                status = initial_status[speaker]
                status_data.append([
                    speaker,
                    "✅" if status['available'] else "❌",
                    status['duration'],
                    status['sample_rate']
                ])
            
            status_table = gr.Dataframe(
                value=status_data,
                headers=["Speaker", "Status", "Duration", "Sample Rate"],
                row_count=4
            )
        
        with gr.Column(scale=2):
            # Text Input
            gr.Markdown("### 📝 Enter Text")
            text_input = gr.Textbox(
                placeholder="Type or paste your text here...",
                lines=5
            )
            
            with gr.Row():
                sample_btn = gr.Button("📋 Sample Text")
                clear_btn = gr.Button("🗑️ Clear")
            
            # Generate Button
            generate_btn = gr.Button("🔊 Generate Speech", variant="primary")
            
            # Status Output
            status_output = gr.Markdown("")
            
            # Audio Output
            gr.Markdown("### 🎧 Generated Audio")
            audio_output = gr.Audio(
                type="filepath",
                label=""
            )
            
            # Hidden fields for filename
            filename_output = gr.Textbox(visible=False)
    
    # Event Handlers
    voice_dropdown.change(
        fn=update_voice_info,
        inputs=[voice_dropdown],
        outputs=[voice_info]
    )
    
    refresh_btn.click(
        fn=refresh_voices,
        inputs=[],
        outputs=[voice_dropdown]
    )
    
    sample_btn.click(
        fn=load_sample_text,
        inputs=[],
        outputs=[text_input]
    )
    
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[text_input]
    )
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[voice_dropdown, text_input],
        outputs=[audio_output, status_output, filename_output]
    )
    
    # Examples
    gr.Markdown("### 📚 Example Texts")
    gr.Examples(
        examples=[
            ["Hello, this is a test of my voice cloning project."],
            ["The quick brown fox jumps over the lazy dog."],
            ["Welcome to my voice cloning demonstration."],
            ["This sounds just like me! The similarity is amazing."]
        ],
        inputs=[text_input]
    )

if __name__ == "__main__":
    # Check if voices are registered
    if not available_voices:
        print("\n" + "="*60)
        print("⚠️  No voices registered!")
        print("Please run: python scripts/register_voices.py")
        print("Then restart this app.")
        print("="*60 + "\n")
    
    # Launch the app
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )