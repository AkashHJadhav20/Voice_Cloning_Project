# ultimate_voice_enhancer.py
"""
ULTIMATE VOICE ENHANCEMENT SUITE
Automatically detects and enhances all voice files with professional audio processing
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing optional advanced libraries
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("⚠️  For better noise reduction: pip install noisereduce")

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False
    print("⚠️  For professional loudness: pip install pyloudnorm")

try:
    from scipy import signal, ndimage
    from scipy.ndimage import median_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class UltimateVoiceEnhancer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.voices_dir = self.project_root / "voices"
        self.enhanced_dir = self.project_root / "voices_enhanced"
        self.enhanced_dir.mkdir(exist_ok=True)
        
        # Professional audio parameters
        self.target_sr = 24000  # Optimal for XTTS
        self.target_loudness = -23.0  # LUFS (broadcast standard)
        self.target_duration = 25  # seconds (optimal for cloning)
        self.min_duration = 15
        self.max_duration = 30
        
        # Detection parameters
        self.speakers = ['AB', 'AP', 'ME', 'DF']
        
    def find_all_audio_files(self):
        """Automatically detect all audio files in the project"""
        audio_files = []
        
        # Search patterns
        search_paths = [
            self.voices_dir,
            self.project_root,
            self.project_root / "audio",
            self.project_root / "recordings",
        ]
        
        # File extensions to look for
        extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
        
        print("\n🔍 SCANNING FOR AUDIO FILES...")
        print("="*60)
        
        for search_path in search_paths:
            if search_path.exists():
                for ext in extensions:
                    found = list(search_path.rglob(f"*{ext}"))
                    for file in found:
                        # Check if it might be one of our speakers
                        for speaker in self.speakers:
                            if speaker.lower() in file.stem.lower():
                                audio_files.append({
                                    'path': file,
                                    'speaker': speaker,
                                    'format': ext
                                })
                                print(f"  ✅ Found {speaker}: {file}")
                                break
        
        # Also look for any files in speaker directories
        for speaker in self.speakers:
            speaker_dir = self.voices_dir / speaker
            if speaker_dir.exists():
                for ext in extensions:
                    for file in speaker_dir.glob(f"*{ext}"):
                        if file not in [f['path'] for f in audio_files]:
                            audio_files.append({
                                'path': file,
                                'speaker': speaker,
                                'format': ext
                            })
                            print(f"  ✅ Found {speaker}: {file}")
        
        print(f"\n📊 Total audio files found: {len(audio_files)}")
        return audio_files
    
    def advanced_noise_profile(self, audio, sr):
        """Create adaptive noise profile from silent parts"""
        # Find silent parts (assume first 0.5s or quietest parts)
        energy = librosa.feature.rms(y=audio)[0]
        silent_frames = np.where(energy < np.percentile(energy, 10))[0]
        
        if len(silent_frames) > 0:
            hop_length = 512
            silent_samples = []
            for frame in silent_frames[:10]:  # Use first few silent frames
                start = frame * hop_length
                end = start + hop_length
                if end < len(audio):
                    silent_samples.append(audio[start:end])
            
            if silent_samples:
                noise_sample = np.concatenate(silent_samples)
                return noise_sample
        return None
    
    def spectral_gate(self, audio, sr, threshold_db=-40):
        """Apply spectral gating for noise reduction"""
        # Compute STFT
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Compute noise floor
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Apply gate
        mask = magnitude > (noise_floor * 10**(threshold_db/20))
        magnitude_clean = magnitude * mask
        
        # Reconstruct
        D_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(D_clean)
        
        return audio_clean
    
    def remove_electrical_interference(self, audio, sr):
        """Remove 50/60 Hz hum and harmonics"""
        # Notch filter for power line noise
        from scipy import signal
        
        # 50 Hz and harmonics (common in some regions)
        freqs_to_remove = [50, 100, 150, 200, 250]
        
        for freq in freqs_to_remove:
            if freq < sr/2:  # Nyquist limit
                b, a = signal.iirnotch(freq, 30, sr)
                audio = signal.filtfilt(b, a, audio)
        
        return audio
    
    def enhance_voice(self, input_path, speaker):
        """Apply comprehensive enhancement pipeline"""
        print(f"\n🎤 ENHANCING: {speaker}")
        print("="*60)
        print(f"📂 Input: {input_path}")
        
        # Step 1: Load and convert to optimal sample rate
        print("1️⃣ Loading and resampling...")
        audio, orig_sr = librosa.load(input_path, sr=None)
        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
            sr = self.target_sr
        else:
            sr = orig_sr
        
        original_duration = len(audio) / sr
        print(f"   Duration: {original_duration:.1f}s @ {sr}Hz")
        
        # Step 2: Remove DC offset
        print("2️⃣ Removing DC offset...")
        audio = audio - np.mean(audio)
        
        # Step 3: Advanced noise reduction
        print("3️⃣ Applying spectral noise reduction...")
        
        # Try multiple noise reduction techniques
        if HAS_NOISEREDUCE:
            # Deep noise reduction
            noise_sample = self.advanced_noise_profile(audio, sr)
            if noise_sample is not None:
                audio = nr.reduce_noise(
                    y=audio, 
                    sr=sr, 
                    y_noise=noise_sample,
                    prop_decrease=0.9,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512,
                    stationary=False
                )
            else:
                audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.85)
        
        # Step 4: Remove electrical interference
        print("4️⃣ Removing electrical interference...")
        audio = self.remove_electrical_interference(audio, sr)
        
        # Step 5: Apply spectral gating
        print("5️⃣ Applying spectral gate...")
        audio = self.spectral_gate(audio, sr, threshold_db=-35)
        
        # Step 6: De-essing (reduce sibilance)
        print("6️⃣ De-essing...")
        # High-frequency reduction for sibilance
        sos = signal.butter(10, 6000, 'lowpass', fs=sr, output='sos')
        audio_low = signal.sosfilt(sos, audio)
        audio_high = audio - audio_low
        audio_high = audio_high * 0.5  # Reduce sibilance
        audio = audio_low + audio_high
        
        # Step 7: Remove silences and keep best parts
        print("7️⃣ Analyzing and selecting best segments...")
        
        # Detect speech segments
        non_silent = librosa.effects.split(audio, top_db=25, frame_length=2048, hop_length=512)
        
        if len(non_silent) > 1:
            print(f"   Found {len(non_silent)} speech segments")
            
            # Calculate quality score for each segment
            best_segments = []
            for i, (start, end) in enumerate(non_silent):
                segment = audio[start:end]
                duration = (end - start) / sr
                
                if duration < 0.5:  # Skip very short segments
                    continue
                
                # Quality metrics
                energy = np.mean(segment**2)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                zero_crossings = np.mean(librosa.feature.zero_crossing_rate(segment))
                
                # Score (higher is better)
                score = energy * 10 + spectral_centroid/1000 - zero_crossings*100
                best_segments.append((score, start, end, segment))
            
            # Sort by quality
            best_segments.sort(reverse=True)
            
            # Select top segments totaling target duration
            selected_audio = []
            total_selected = 0
            target_samples = self.target_duration * sr
            
            for score, start, end, segment in best_segments:
                if total_selected + len(segment) <= target_samples:
                    selected_audio.append(segment)
                    total_selected += len(segment)
                    print(f"   ✓ Selected segment {len(selected_audio)}: {len(segment)/sr:.1f}s (quality: {score:.2f})")
            
            if selected_audio:
                # Add small pauses between segments
                pause = np.zeros(int(0.1 * sr))
                audio = np.concatenate([seg for seg in selected_audio[:-1] for _ in (seg, pause)] + [selected_audio[-1]])
            else:
                # Fallback: take middle portion
                mid = len(audio) // 2
                half_target = (self.target_duration * sr) // 2
                start = max(0, mid - half_target)
                end = min(len(audio), mid + half_target)
                audio = audio[start:end]
        else:
            # Single continuous segment - take middle portion
            if len(audio) > self.target_duration * sr:
                mid = len(audio) // 2
                half_target = (self.target_duration * sr) // 2
                start = max(0, mid - half_target)
                end = min(len(audio), mid + half_target)
                audio = audio[start:end]
        
        # Step 8: Professional loudness normalization
        print("8️⃣ Applying professional loudness normalization...")
        
        if HAS_PYLOUDNORM:
            # EBU R128 loudness normalization
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, self.target_loudness)
        else:
            # Fallback: RMS normalization
            current_rms = np.sqrt(np.mean(audio**2))
            target_rms = 0.1
            audio = audio * (target_rms / (current_rms + 1e-10))
        
        # Step 9: Dynamic range compression
        print("9️⃣ Applying gentle compression...")
        
        # Simple compression
        threshold = 0.1
        ratio = 3.0
        gain = 1.0
        
        mask = np.abs(audio) > threshold
        audio_compressed = audio.copy()
        audio_compressed[mask] = threshold + (np.abs(audio[mask]) - threshold) / ratio
        audio_compressed[mask] *= np.sign(audio[mask])
        audio = audio_compressed * gain
        
        # Step 10: Final peak normalization
        print("🔟 Final peak normalization...")
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.95
        
        # Step 11: Fade in/out to prevent clicks
        print("   Applying fades...")
        fade_len = int(0.01 * sr)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        audio[:fade_len] *= fade_in
        audio[-fade_len:] *= fade_out
        
        # Final stats
        final_duration = len(audio) / sr
        final_rms = np.sqrt(np.mean(audio**2))
        final_peak = np.max(np.abs(audio))
        
        print("\n📊 ENHANCEMENT RESULTS")
        print("-"*60)
        print(f"   Original duration: {original_duration:.1f}s")
        print(f"   Final duration:    {final_duration:.1f}s")
        print(f"   Original RMS:      {np.sqrt(np.mean((audio_pre_enhance if 'audio_pre_enhance' in locals() else audio)**2)):.3f}")
        print(f"   Final RMS:         {final_rms:.3f}")
        print(f"   Peak level:        {final_peak:.3f}")
        print(f"   Sample rate:       {sr}Hz")
        
        return audio, sr
    
    def process_all_voices(self):
        """Main processing function"""
        print("\n" + "="*70)
        print("🎤 ULTIMATE VOICE ENHANCEMENT SUITE")
        print("="*70)
        
        # Find all audio files
        audio_files = self.find_all_audio_files()
        
        if not audio_files:
            print("\n❌ No audio files found!")
            print("\nPlease place your voice files in one of these locations:")
            print("  • voices/AB/ (any audio file)")
            print("  • voices/AP/ (any audio file)")
            print("  • voices/ME/ (any audio file)")
            print("  • voices/DF/ (any audio file)")
            print("  • Project root as AB.wav, AP.wav, etc.")
            return
        
        # Process each speaker
        enhanced_files = []
        
        for file_info in audio_files:
            try:
                # Enhance audio
                audio, sr = self.enhance_voice(file_info['path'], file_info['speaker'])
                
                # Save enhanced version
                output_dir = self.voices_dir / file_info['speaker']
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / "reference_ultimate.wav"
                
                sf.write(output_path, audio, sr)
                enhanced_files.append((file_info['speaker'], output_path))
                
                print(f"\n✅ Enhanced {file_info['speaker']} saved to: {output_path}")
                
            except Exception as e:
                print(f"\n❌ Error processing {file_info['speaker']}: {e}")
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ENHANCEMENT COMPLETE!")
        print("="*70)
        
        print("\n📁 Enhanced files created:")
        for speaker, path in enhanced_files:
            print(f"  • {speaker}: {path}")
        
        print("\n📌 NEXT STEPS:")
        print("  1. Review the enhanced files (play them to check quality)")
        print("  2. Replace originals with enhanced versions:")
        for speaker, path in enhanced_files:
            print(f'     Move-Item "{path}" "voices/{speaker}/reference.wav" -Force')
        print("  3. Re-register voices:")
        print("     python scripts/register_voices.py")
        print("\n   Or use this command to do it all at once:")
        print("   (ONLY if you're satisfied with the enhancements)")
        for speaker, path in enhanced_files:
            print(f'     Move-Item "{path}" "voices/{speaker}/reference.wav" -Force')
        print("     python scripts/register_voices.py")
        print("="*70)

def main():
    # Install required packages if missing
    required_packages = ['noisereduce', 'pyloudnorm', 'librosa', 'soundfile']
    
    print("Checking for required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Run enhancer
    enhancer = UltimateVoiceEnhancer()
    enhancer.process_all_voices()

if __name__ == "__main__":
    main()