# == KEYS ==
OPENAI_API = ""

# == OPTIONS ==
LLM_PROVIDER = "openai" # or "huggingface"
# if huggingface:
HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HF_DEVICE = "auto" # cpu, cuda, etc
# if openai
OPENAI_MODEL = "gpt-4o-mini"

ASR_MODEL = "base" # whisper models --> tiny, base, small, medium, large
ASR_DEVICE = "cpu" # cpu, cuda, etc

TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"