import sounddevice as sd
import numpy as np

def play_audio(audio_data, sample_rate=44100):

    if audio_data is None or len(audio_data) == 0:
        print("Error: No audio data to play")
        return

    audio_data = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    audio_data_peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0

    # Normalize audio if needed
    if audio_data_peak > 1.0:
        audio_data = audio_data / audio_data_peak

    sd.stop()
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

def stop_audio():
    sd.stop()