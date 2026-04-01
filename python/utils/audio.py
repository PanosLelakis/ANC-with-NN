import sounddevice as sd
import numpy as np
import threading

_AUDIO_LOCK = threading.Lock()

def play_audio(audio_data, sample_rate=44100):
    if audio_data is None or len(audio_data) == 0:
        print("Error: No audio data to play")
        return

    audio_data = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
    if peak > 1.0:
        audio_data = audio_data / peak

    with _AUDIO_LOCK:
        sd.stop()
        sd.play(audio_data, samplerate=sample_rate, blocking=False)

def is_audio_active():
    try:
        s = sd.get_stream()
        return bool(s is not None and s.active)
    except Exception:
        return False

def stop_audio():
    with _AUDIO_LOCK:
        sd.stop()

def save_wav(before_signal, after_signal, fs, base):
    from scipy.io import wavfile
    import os
    
    # Convert to numpy arrays float32
    fs = int(fs)
    before = np.asarray(before_signal, dtype=np.float32)
    after = np.asarray(after_signal, dtype=np.float32)
    after = np.nan_to_num(after, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale signal to avoid clipping, based on max abs value of either signals
    max_abs = max(float(np.max(np.abs(before))), float(np.max(np.abs(after))), 1e-6)
    scale = min(1.0, 0.99 / max_abs)
    #before = np.clip(before * scale, -1.0, 1.0)
    after = np.clip(after * scale, -1.0, 1.0)

    #before_i16 = (before * 32767.0).astype(np.int16)
    after_i16 = (after * 32767.0).astype(np.int16)

    #wavfile.write(os.path.join(base, "input.wav"), fs, before_i16)
    wavfile.write(os.path.join(base, "output.wav"), fs, after_i16)