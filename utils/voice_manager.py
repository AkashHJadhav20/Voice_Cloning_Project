# utils/voice_manager.py
import json
from pathlib import Path
import logging
from datetime import datetime
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class VoiceManager:
    def __init__(self, voices_dir="voices"):
        self.voices_dir = Path(voices_dir)
        self.registry_file = self.voices_dir / "voice_registry.json"
        self.audio_processor = AudioProcessor()
        self.voice_registry = {}
        self.load_registry()
        
    def load_registry(self):
        """Load voice registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.voice_registry = json.load(f)
                logger.info(f"Loaded {len(self.voice_registry)} voices from registry")
                return True
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
        return False
    
    def save_registry(self):
        """Save voice registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.voice_registry, f, indent=2)
            logger.info("Registry saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            return False
    
    def add_voice(self, speaker_name, audio_path):
        """Add a new voice to registry"""
        try:
            # Validate audio
            is_valid, message = self.audio_processor.validate_audio(audio_path)
            if not is_valid:
                return False, message
            
            # Get audio info
            audio, sr = self.audio_processor.load_audio(audio_path)
            duration = len(audio) / sr
            
            # Add to registry
            self.voice_registry[speaker_name] = {
                "audio_path": str(audio_path),
                "speaker_name": speaker_name,
                "duration": duration,
                "sample_rate": sr,
                "date_added": datetime.now().isoformat(),
                "is_active": True
            }
            
            self.save_registry()
            return True, f"Voice added: {duration:.1f}s"
            
        except Exception as e:
            logger.error(f"Error adding voice: {e}")
            return False, str(e)
    
    def remove_voice(self, speaker_name):
        """Remove a voice from registry"""
        if speaker_name in self.voice_registry:
            del self.voice_registry[speaker_name]
            self.save_registry()
            return True
        return False
    
    def get_voice(self, speaker_name):
        """Get voice info by name"""
        return self.voice_registry.get(speaker_name)
    
    def get_voice_path(self, speaker_name):
        """Get audio file path for a voice"""
        voice = self.get_voice(speaker_name)
        if voice:
            path = voice.get("audio_path")
            if Path(path).exists():
                return path
        return None
    
    def list_voices(self):
        """List all registered voices"""
        return list(self.voice_registry.keys())
    
    def get_stats(self):
        """Get statistics about registered voices"""
        return {
            "total_voices": len(self.voice_registry),
            "voice_names": self.list_voices(),
            "expected_speakers": ["AB", "AP", "ME", "DF"]
        }