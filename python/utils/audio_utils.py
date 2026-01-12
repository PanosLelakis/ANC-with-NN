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

    # Serialize start (stop+play) so two threads don't fight
    with _AUDIO_LOCK:
        sd.stop()
        sd.play(audio_data, samplerate=sample_rate)

    # Do NOT hold the lock during wait, so stop_audio can run
    sd.wait()

def stop_audio():
    with _AUDIO_LOCK:
        sd.stop()