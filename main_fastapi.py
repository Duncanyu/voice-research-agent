from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
from src.asr.transcriber import transcribe
from src.llm.openai_llm import gpt_respond
from src.tts.speak import generate_tts
from config import OPENAI_API, OPENAI_MODEL, TTS_MODEL, TTS_SPEAKER

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("static/index.html").read()

@app.post("/chat/")
async def chat(audio_file: UploadFile = File(...)):
    wav_bytes = await audio_file.read()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(wav_bytes)
        temp_path = tmp.name

    prompt = transcribe(temp_path)
    response_text = gpt_respond(prompt, OPENAI_API, OPENAI_MODEL)
    tts_path = generate_tts(response_text, TTS_MODEL, TTS_SPEAKER)

    return Response(open(tts_path, 'rb').read(), media_type='audio/wav')

if __name__ == '__main__':
    uvicorn.run('main_fastapi:app', host='0.0.0.0', port=8000, reload=True)