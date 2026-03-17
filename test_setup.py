# test_setup.py
import torch
import TTS
import soundfile
import librosa
import numpy as np
import sys
from pathlib import Path

print("\n" + "="*60)
print("VOICE CLONING PROJECT - SETUP VERIFICATION")
print("="*60)

# Test Python version
print(f"\n📌 Python: {sys.version}")

# Test PyTorch
print(f"\n📌 PyTorch: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Test TTS
print(f"\n📌 TTS: {TTS.__version__}")

# Test other libraries
print(f"\n📌 NumPy: {np.__version__}")
print(f"📌 SoundFile: {soundfile.__version__}")
print(f"📌 Librosa: {librosa.__version__}")

# Check voice directories
print("\n📌 Voice Files:")
voices_dir = Path("voices")
speakers = ["AB", "AP", "ME", "DF"]

for speaker in speakers:
    wav_path = voices_dir / speaker / "reference.wav"
    if wav_path.exists():
        size = wav_path.stat().st_size / 1024
        print(f"   ✓ {speaker}: {wav_path} ({size:.0f}KB)")
    else:
        alt_path = voices_dir / f"{speaker}.wav"
        if alt_path.exists():
            size = alt_path.stat().st_size / 1024
            print(f"   ✓ {speaker}: {alt_path} ({size:.0f}KB)")
        else:
            print(f"   ✗ {speaker}: Not found")

print("\n" + "="*60)
print("✅ Setup is ready!")
print("="*60 + "\n")