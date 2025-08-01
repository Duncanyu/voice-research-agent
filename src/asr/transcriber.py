from faster_whisper import WhisperModel

def transcribe(audio_path, model_size = "small", device = "cpu"):
    model = WhisperModel(model_size, device=device)

    print(f"transcribing: {audio_path}")
    segments, info = model.transcribe(audio_path)

    text = " ".join([segment.text for segment in segments])

    return text