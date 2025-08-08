import sounddevice as sd
import numpy as np
import tempfile
import wave
import webrtcvad
import time
from collections import deque

def save_wav(audio_data: np.ndarray, sample_rate: int = 16000):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return temp_file.name

def record(sample_rate: int = 16000, frame_duration: int = 30, padding_duration: float = 1.0, vad_aggressiveness: int = 3, max_wait_for_speech: float = 2.0,):
    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(sample_rate * frame_duration / 1000)

    start_time = time.time()
    last_speech_time = None
    speech_started = False

    all_frames: list[bytes] = []
    silence_buffer = deque(maxlen=int(padding_duration * 1000 / frame_duration))

    def callback(indata, frames, time_info, status):
        nonlocal speech_started, last_speech_time

        frame = bytes(indata)
        all_frames.append(frame)

        now = time.time()
        is_speech = vad.is_speech(frame, sample_rate)

        if not speech_started:
            if is_speech:
                speech_started = True
                last_speech_time = now
                print("Speech detected.", flush=True)
            else:
                if now - start_time > max_wait_for_speech:
                    print("No speech in time, stopping.", flush=True)
                    raise sd.CallbackStop()
        else:
            if is_speech:
                last_speech_time = now
                silence_buffer.clear()
            else:
                silence_buffer.append(frame)
                if now - (last_speech_time or now) > padding_duration:
                    print("Silence end, stopping recording.", flush=True)
                    raise sd.CallbackStop()

    with sd.RawInputStream(samplerate = sample_rate, blocksize = frame_size, dtype = 'int16', channels = 1, callback = callback) as stream:
        print("Recording...")
        while stream.active:
            time.sleep(0.01)

    if not speech_started:
        # no speech ever detected
        return None

    audio = np.frombuffer(b''.join(all_frames), dtype=np.int16)
    path = save_wav(audio, sample_rate)
    print(f"Saved recording to {path}", flush=True)
    return path
