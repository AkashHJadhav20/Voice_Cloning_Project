import torch
import transformers
import TTS
import sys

print("="*50)
print("VERSION COMPATIBILITY CHECK")
print("="*50)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"TTS: {TTS.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Check compatibility
if torch.__version__ >= "2.4":
    print("✓ PyTorch version is compatible")
else:
    print("⚠ PyTorch < 2.4 - May have issues with latest TTS")

if transformers.__version__ >= "4.36":
    print("✓ Transformers version is compatible")
else:
    print("⚠ Transformers version may be too old")

print("="*50)