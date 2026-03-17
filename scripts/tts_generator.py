import os
import sys
import torch
import time
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from TTS.api import TTS
    from utils.voice_manager import VoiceManager
    print("✓ Modules imported successfully")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)

class TTSGenerator:
    def __init__(self):
        print("\n" + "="*60)
        print("INITIALIZING TTS GENERATOR")
        print("="*60)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📌 Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"📌 GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
        
        print("\n📌 Loading XTTS-v2 model...")
        try:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
        
        self.voice_manager = VoiceManager()
        self.output_dir = Path("generated_audio")
        self.output_dir.mkdir(exist_ok=True)
        
        self.current_voice = None
        self.speakers = ["AB", "AP", "ME", "DF"]
        
        print("="*60)
    
    def list_voices(self):
        registered = self.voice_manager.list_voices()
        print("\n" + "="*60)
        print("AVAILABLE VOICES")
        print("="*60)
        
        for speaker in self.speakers:
            status = "✓" if speaker in registered else "✗"
            path = self.voice_manager.get_voice_path(speaker)
            if path and os.path.exists(path):
                print(f"  {status} {speaker} - Ready")
            else:
                print(f"  {status} {speaker} - Not registered")
        
        print("="*60)
        return registered
    
    def select_voice(self):
        registered = self.voice_manager.list_voices()
        
        if not registered:
            print("\n✗ No voices registered! Please run register_voices.py first.")
            return False
        
        print("\n🎤 Select a voice:")
        for i, speaker in enumerate(self.speakers, 1):
            if speaker in registered:
                print(f"  {i}. {speaker}")
        
        while True:
            try:
                choice = input("\nEnter voice name or number: ").strip()
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.speakers):
                        selected = self.speakers[idx]
                        if selected in registered:
                            self.current_voice = selected
                            print(f"\n✓ Voice set to: {self.current_voice}")
                            return True
                elif choice.upper() in self.speakers:
                    if choice.upper() in registered:
                        self.current_voice = choice.upper()
                        print(f"\n✓ Voice set to: {self.current_voice}")
                        return True
                elif choice.lower() == 'quit':
                    return False
                else:
                    print("Invalid choice. Try again.")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                sys.exit(0)
    
    def generate(self, text):
        if not self.current_voice:
            print("✗ No voice selected!")
            return None
        
        speaker_wav = self.voice_manager.get_voice_path(self.current_voice)
        if not speaker_wav or not os.path.exists(speaker_wav):
            print(f"✗ Voice file for {self.current_voice} not found!")
            return None
        
        timestamp = int(time.time())
        output_path = self.output_dir / f"{self.current_voice}_{timestamp}.wav"
        
        print(f"\n🎤 Generating: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            start = time.time()
            
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="en",
                file_path=str(output_path)
            )
            
            elapsed = time.time() - start
            
            print(f"\n✓ Generated: {output_path}")
            print(f"  Time: {elapsed:.1f}s")
            
            return output_path
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def run(self):
        print("\n" + "="*60)
        print("🎤 TTS GENERATOR - VOICE CLONING")
        print("="*60)
        
        registered = self.list_voices()
        if not registered:
            input("\nPress Enter to exit...")
            return
        
        if not self.select_voice():
            return
        
        print("\n" + "-"*60)
        print("Commands: 'list', 'switch', 'quit'")
        print("-"*60)
        
        while True:
            try:
                text = input(f"\n[{self.current_voice}] > ").strip()
                
                if text.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                elif text.lower() == 'list':
                    self.list_voices()
                elif text.lower() == 'switch':
                    self.current_voice = None
                    self.select_voice()
                elif text:
                    self.generate(text)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

def main():
    try:
        generator = TTSGenerator()
        generator.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
