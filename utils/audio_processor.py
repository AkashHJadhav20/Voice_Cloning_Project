# utils/audio_processor.py
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, target_sr=24000):
        self.target_sr = target_sr
        
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            audio, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            return None, None
    
    def validate_audio(self, file_path, min_duration=10, max_duration=60):
        """Validate audio file for voice cloning"""
        try:
            # Check file exists
            if not Path(file_path).exists():
                return False, "File not found"
            
            # Check file extension
            if not str(file_path).lower().endswith('.wav'):
                return False, "File must be WAV format"
            
            # Load audio
            audio, sr = self.load_audio(file_path)
            if audio is None:
                return False, "Could not load audio"
            
            # Check duration
            duration = len(audio) / sr
            if duration < min_duration:
                return False, f"Audio too short: {duration:.1f}s (min {min_duration}s)"
            if duration > max_duration:
                return False, f"Audio too long: {duration:.1f}s (max {max_duration}s)"
            
            # Check volume
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                return False, "Audio too quiet"
            if rms > 0.95:
                return False, "Audio clipping (too loud)"
            
            return True, f"Valid audio: {duration:.1f}s, {sr}Hz"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def trim_silence(self, audio, sr, top_db=20):
        """Trim silence from beginning and end"""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed