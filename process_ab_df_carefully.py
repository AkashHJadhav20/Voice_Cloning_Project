import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
import noisereduce as nr
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def professional_enhancement(input_path, output_path, target_duration=25):
    """Apply same professional enhancement as ME/AP without over-trimming"""
    
    print(f"\n🎤 Processing: {input_path.parent.name}")
    print("="*60)
    
    # Step 1: Load audio
    print("1️⃣ Loading audio...")
    audio, sr = librosa.load(input_path, sr=24000)
    original_duration = len(audio) / sr
    print(f"   Original: {original_duration:.1f}s @ {sr}Hz")
    
    # Store original for comparison
    audio_original = audio.copy()
    
    # Step 2: Advanced noise reduction (same as ME/AP)
    print("2️⃣ Applying spectral noise reduction...")
    try:
        # Create noise profile from quiet parts
        noise_sample = audio[:int(0.5 * sr)]  # First 0.5s as noise sample
        audio = nr.reduce_noise(
            y=audio, 
            sr=sr, 
            y_noise=noise_sample,
            prop_decrease=0.85,
            stationary=False
        )
    except:
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    
    # Step 3: Remove electrical interference (50/60 Hz hum)
    print("3️⃣ Removing electrical interference...")
    # Notch filter at 50Hz and harmonics
    for freq in [50, 100, 150, 200]:
        if freq < sr/2:
            b, a = signal.iirnotch(freq, 30, sr)
            audio = signal.filtfilt(b, a, audio)
    
    # Step 4: Gentle high-pass filter (remove rumble)
    print("4️⃣ Applying high-pass filter...")
    sos = signal.butter(10, 80, 'hp', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # Step 5: Intelligent silence preservation (NOT removal)
    print("5️⃣ Analyzing speech pattern...")
    
    # Detect speech segments (but we'll KEEP them all, just note them)
    non_silent = librosa.effects.split(audio, top_db=25, frame_length=2048, hop_length=512)
    
    if len(non_silent) > 1:
        speech_duration = sum((end-start) for start,end in non_silent) / sr
        silence_duration = original_duration - speech_duration
        print(f"   Speech: {speech_duration:.1f}s, Silence: {silence_duration:.1f}s")
        print(f"   Keeping ALL segments ({len(non_silent)} segments)")
    
    # Step 6: Professional loudness normalization (same as ME/AP)
    print("6️⃣ Normalizing loudness...")
    
    # Calculate current loudness
    current_rms = np.sqrt(np.mean(audio**2))
    target_rms = 0.12  # Optimal level for XTTS
    
    # Apply gentle compression first
    threshold = 0.1
    ratio = 2.5
    
    # Soft compression
    mask = np.abs(audio) > threshold
    audio_compressed = audio.copy()
    audio_compressed[mask] = threshold + (np.abs(audio[mask]) - threshold) / ratio
    audio_compressed[mask] *= np.sign(audio[mask])
    audio = audio_compressed
    
    # Normalize to target RMS
    current_rms = np.sqrt(np.mean(audio**2))
    gain = target_rms / (current_rms + 1e-10)
    audio = audio * gain
    
    # Step 7: De-essing (reduce sibilance) - same as ME/AP
    print("7️⃣ De-essing...")
    # Split into low and high frequencies
    sos_low = signal.butter(10, 6000, 'lowpass', fs=sr, output='sos')
    audio_low = signal.sosfilt(sos_low, audio)
    audio_high = audio - audio_low
    audio_high = audio_high * 0.6  # Reduce sibilance
    audio = audio_low + audio_high
    
    # Step 8: Final peak normalization
    print("8️⃣ Final normalization...")
    audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.95
    
    # Step 9: Add gentle fades (preserves all content)
    print("9️⃣ Adding gentle fades...")
    fade_len = int(0.01 * sr)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    
    # Step 10: Duration management (KEEP ORIGINAL DURATION, just ensure it's within limits)
    final_duration = len(audio) / sr
    print(f"🔟 Final duration: {final_duration:.1f}s")
    
    # Only trim if ABSOLUTELY necessary (over 60s for XTTS limit)
    if final_duration > 55:  # XTTS max is 60s, so 55s is safe
        # Take middle portion to preserve quality
        samples_to_keep = int(50 * sr)  # Keep 50 seconds
        start = (len(audio) - samples_to_keep) // 2
        audio = audio[start:start + samples_to_keep]
        print(f"   Slight trim to {len(audio)/sr:.1f}s (within XTTS limits)")
    
    # Calculate final stats
    final_duration = len(audio) / sr
    final_rms = np.sqrt(np.mean(audio**2))
    final_peak = np.max(np.abs(audio))
    
    print("\n📊 ENHANCEMENT RESULTS")
    print("-"*60)
    print(f"   Original duration: {original_duration:.1f}s")
    print(f"   Final duration:    {final_duration:.1f}s")
    print(f"   Duration change:   {final_duration - original_duration:+.1f}s")
    print(f"   Original RMS:      {np.sqrt(np.mean(audio_original**2)):.3f}")
    print(f"   Final RMS:         {final_rms:.3f}")
    print(f"   Peak level:        {final_peak:.3f}")
    print(f"   Segments preserved: {len(non_silent)}")
    print(f"   ✅ All original content preserved!" if abs(final_duration - original_duration) < 1 else f"   ⚠ Duration adjusted for compatibility")
    
    # Save enhanced version
    sf.write(output_path, audio, sr)
    print(f"\n💾 Saved to: {output_path}")
    
    return {
        'speaker': input_path.parent.name,
        'original_duration': original_duration,
        'final_duration': final_duration,
        'original_rms': np.sqrt(np.mean(audio_original**2)),
        'final_rms': final_rms
    }

def main():
    print("="*70)
    print("🎤 PROFESSIONAL VOICE ENHANCEMENT FOR AB & DF")
    print("="*70)
    print("\nApplying same enhancement as ME/AP but preserving all content")
    print("-"*70)
    
    # Process only AB and DF
    speakers_to_process = ['AB', 'DF']
    base_path = Path("voices")
    results = []
    
    for speaker in speakers_to_process:
        input_path = base_path / speaker / "reference.wav"
        
        if not input_path.exists():
            print(f"\n❌ No reference.wav found for {speaker}")
            continue
        
        # Create backup first
        backup_path = base_path / speaker / "reference_backup.wav"
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"\n💾 Created backup: {backup_path}")
        
        # Process the file (save as temporary first)
        temp_output = base_path / speaker / "reference_enhanced_temp.wav"
        stats = professional_enhancement(input_path, temp_output)
        results.append(stats)
    
    # Summary
    print("\n" + "="*70)
    print("📊 ENHANCEMENT SUMMARY")
    print("="*70)
    for stat in results:
        print(f"\n{stat['speaker']}:")
        print(f"  Duration: {stat['original_duration']:.1f}s → {stat['final_duration']:.1f}s")
        print(f"  RMS: {stat['original_rms']:.3f} → {stat['final_rms']:.3f}")
    
    print("\n" + "="*70)
    print("✅ ENHANCEMENT COMPLETE!")
    print("="*70)
    print("\n📌 NEXT STEPS:")
    print("  1. Review the enhanced files (play them to verify quality)")
    print("  2. If satisfied, replace originals:")
    for speaker in speakers_to_process:
        print(f'     Move-Item "voices/{speaker}/reference_enhanced_temp.wav" "voices/{speaker}/reference.wav" -Force')
    print("  3. Re-register voices:")
    print("     python scripts/register_voices.py")
    print("\n   Or run this command to do it now:")
    print("   (Only if you're satisfied with the enhancement)")
    for speaker in speakers_to_process:
        print(f'     Move-Item "voices/{speaker}/reference_enhanced_temp.wav" "voices/{speaker}/reference.wav" -Force')
    print("     python scripts/register_voices.py")
    print("="*70)

if __name__ == "__main__":
    main()