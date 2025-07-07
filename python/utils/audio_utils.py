import sounddevice as sd
import numpy as np

# Global state to track playback
is_playing = False

def play_audio(audio_data, sample_rate=44100):
    global is_playing

    if audio_data is None or len(audio_data) == 0:
        print("Error: No audio data to play")
        return

    # Normalize audio if needed
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    is_playing = True
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()
    is_playing = False

def stop_audio():
    global is_playing
    sd.stop()
    is_playing = False