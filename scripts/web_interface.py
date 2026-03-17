# scripts/web_interface.py
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template_string, request, jsonify, send_file
import torch
import uuid
from TTS.api import TTS
from utils.voice_manager import VoiceManager

app = Flask(__name__)
voice_manager = VoiceManager()
output_dir = Path("generated_audio")
output_dir.mkdir(exist_ok=True)

# Initialize TTS
print("Loading TTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"✓ Model loaded on {device}")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voice Cloning TTS</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f2f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1a73e8;
            margin-top: 0;
            text-align: center;
        }
        .voice-selector {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        .voice-btn {
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            background: white;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .voice-btn:hover {
            border-color: #1a73e8;
            background: #e8f0fe;
        }
        .voice-btn.active {
            background: #1a73e8;
            color: white;
            border-color: #1a73e8;
        }
        .voice-btn.disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #f5f5f5;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            margin: 20px 0;
            box-sizing: border-box;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #1a73e8;
        }
        .generate-btn {
            width: 100%;
            padding: 15px;
            background: #1a73e8;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.3s;
        }
        .generate-btn:hover {
            background: #1557b0;
        }
        .generate-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .audio-player {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .audio-player h3 {
            margin-top: 0;
            color: #1a73e8;
        }
        audio {
            width: 100%;
            margin-top: 10px;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            display: block;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            display: block;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #666;
        }
        .speaker-status {
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voice Cloning TTS</h1>
        <p style="text-align: center; color: #666;">Speakers: AB, AP, ME, DF</p>
        
        <div class="status" id="status"></div>
        
        <div class="voice-selector">
            {% for speaker in speakers %}
            <button class="voice-btn {% if speaker.ready %}active{% else %}disabled{% endif %}" 
                    onclick="selectVoice('{{ speaker.name }}')"
                    {% if not speaker.ready %}disabled{% endif %}>
                {{ speaker.name }}
                <div class="speaker-status">
                    {% if speaker.ready %}✅ Ready{% else %}❌ Not registered{% endif %}
                </div>
            </button>
            {% endfor %}
        </div>
        
        <textarea id="text" rows="4" placeholder="Enter text to synthesize..."></textarea>
        
        <button class="generate-btn" onclick="generateSpeech()" id="generateBtn">Generate Speech</button>
        
        <div class="audio-player" id="audioPlayer">
            <h3>Generated Audio</h3>
            <audio id="audio" controls>
                <source src="" type="audio/wav">
            </audio>
        </div>
    </div>
    
    <div class="footer">
        <p>Powered by XTTS-v2 | GTX 1650 Optimized</p>
    </div>
    
    <script>
        let selectedVoice = 'AB';
        
        function selectVoice(voice) {
            selectedVoice = voice;
            document.querySelectorAll('.voice-btn').forEach(btn => {
                if (btn.textContent.includes(voice)) {
                    btn.classList.add('active');
                } else if (!btn.disabled) {
                    btn.classList.remove('active');
                }
            });
            showStatus('Selected voice: ' + voice, 'success');
        }
        
        async function generateSpeech() {
            const text = document.getElementById('text').value;
            
            if (!text) {
                showStatus('Please enter some text', 'error');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.textContent = 'Generating...';
            
            showStatus('Generating speech...', 'success');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        voice: selectedVoice,
                        text: text
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const audioPlayer = document.getElementById('audioPlayer');
                    const audio = document.getElementById('audio');
                    audio.src = '/audio/' + data.filename;
                    audioPlayer.style.display = 'block';
                    audio.load();
                    audio.play();
                    showStatus('Speech generated successfully!', 'success');
                } else {
                    showStatus('Error: ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Speech';
            }
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    speakers = []
    for name in ['AB', 'AP', 'ME', 'DF']:
        path = voice_manager.get_voice_path(name)
        ready = path and os.path.exists(path)
        speakers.append({'name': name, 'ready': ready})
    return render_template_string(HTML_TEMPLATE, speakers=speakers)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    voice = data.get('voice')
    text = data.get('text')
    
    if not voice or not text:
        return jsonify({'success': False, 'error': 'Missing voice or text'})
    
    # Get voice path
    speaker_wav = voice_manager.get_voice_path(voice)
    if not speaker_wav or not os.path.exists(speaker_wav):
        return jsonify({'success': False, 'error': f'Voice {voice} not found'})
    
    try:
        filename = f"{voice}_{uuid.uuid4()}.wav"
        output_path = output_dir / filename
        
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=str(output_path)
        )
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(output_dir / filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)