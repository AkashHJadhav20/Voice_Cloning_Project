# scripts/register_voices.py
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.voice_manager import VoiceManager
from utils.audio_processor import AudioProcessor

class VoiceRegistrar:
    def __init__(self):
        self.voice_manager = VoiceManager()
        self.audio_processor = AudioProcessor()
        self.voices_dir = Path("voices")
        
    def list_available_wav_files(self):
        """List all .wav files in voices directory"""
        wav_files = []
        for file in self.voices_dir.glob("*.wav"):
            wav_files.append(file)
        
        # Also check in speaker subdirectories
        for speaker_dir in self.voices_dir.glob("*/"):
            for file in speaker_dir.glob("*.wav"):
                wav_files.append(file)
                
        return wav_files
    
    def suggest_speaker_names(self):
        """Suggest speaker names based on your files"""
        suggestions = {
            "AB.wav": "AB",
            "AP.wav": "AP", 
            "ME.wav": "ME",
            "DF.wav": "DF"
        }
        return suggestions
    
    def interactive_registration(self):
        """Interactive voice registration"""
        print("\n" + "="*60)
        print("VOICE REGISTRATION SYSTEM")
        print("="*60)
        
        # Check for existing voices
        existing = self.voice_manager.list_voices()
        if existing:
            print(f"\nExisting voices: {', '.join(existing)}")
            overwrite = input("\nDo you want to re-register voices? (y/n): ").lower()
            if overwrite != 'y':
                print("Keeping existing voices.")
                return
        
        # Look for WAV files
        wav_files = self.list_available_wav_files()
        suggestions = self.suggest_speaker_names()
        
        print("\n" + "-"*60)
        print("REGISTER YOUR 4 SPEAKERS")
        print("-"*60)
        print("Speaker names: AB, AP, ME, DF")
        print("\nPlease provide the paths to your WAV files:")
        print("(You can drag and drop files into this window)\n")
        
        speakers_to_register = [
            ("AB", "AB.wav"),
            ("AP", "AP.wav"),
            ("ME", "ME.wav"), 
            ("DF", "DF.wav")
        ]
        
        registered = []
        
        for speaker_name, default_filename in speakers_to_register:
            print(f"\n--- Registering: {speaker_name} ---")
            
            # Check if file exists in standard location
            standard_path = self.voices_dir / speaker_name / "reference.wav"
            if standard_path.exists():
                print(f"Found existing file: {standard_path}")
                use_existing = input(f"Use this file for {speaker_name}? (y/n): ").lower()
                if use_existing == 'y':
                    success, message = self.voice_manager.add_voice(speaker_name, str(standard_path))
                    if success:
                        print(f"✓ {speaker_name} registered: {message}")
                        registered.append(speaker_name)
                    else:
                        print(f"✗ Failed: {message}")
                    continue
            
            # Ask for file path
            while True:
                file_path = input(f"Enter path to {speaker_name}'s WAV file: ").strip()
                file_path = file_path.strip('"').strip("'")
                
                if not file_path:
                    print("Skipping...")
                    break
                
                if os.path.exists(file_path):
                    # Validate audio
                    is_valid, message = self.audio_processor.validate_audio(file_path)
                    if is_valid:
                        # Copy to proper location
                        dest_dir = self.voices_dir / speaker_name
                        dest_dir.mkdir(exist_ok=True)
                        dest_path = dest_dir / "reference.wav"
                        
                        import shutil
                        shutil.copy2(file_path, dest_path)
                        
                        # Register
                        success, msg = self.voice_manager.add_voice(speaker_name, str(dest_path))
                        if success:
                            print(f"✓ {speaker_name} registered successfully!")
                            registered.append(speaker_name)
                        else:
                            print(f"✗ Registration failed: {msg}")
                        break
                    else:
                        print(f"✗ Invalid audio: {message}")
                        retry = input("Try another file? (y/n): ").lower()
                        if retry != 'y':
                            break
                else:
                    print(f"File not found: {file_path}")
                    retry = input("Try again? (y/n): ").lower()
                    if retry != 'y':
                        break
        
        # Summary
        print("\n" + "="*60)
        print("REGISTRATION SUMMARY")
        print("="*60)
        print(f"Successfully registered: {', '.join(registered)}")
        if len(registered) < 4:
            print(f"Missing: {', '.join([s for s in ['AB','AP','ME','DF'] if s not in registered])}")
        print("\nVoice registry saved to: voices/voice_registry.json")
        print("="*60)

def main():
    registrar = VoiceRegistrar()
    registrar.interactive_registration()

if __name__ == "__main__":
    main()