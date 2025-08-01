from src.llm.openai_llm import gpt_respond
from src.llm.hf_llm import hf_respond
from src.asr.asr_pipeline import get_prompt

from config import OPENAI_API, OPENAI_MODEL, HF_DEVICE, HF_MODEL_NAME, HF_TOKEN
from config import LLM_PROVIDER

if LLM_PROVIDER == "openai":
    while True:
        print(gpt_respond(get_prompt(), OPENAI_API, OPENAI_MODEL))
elif LLM_PROVIDER == "huggingface":
    while True:
        print(hf_respond(get_prompt(), HF_MODEL_NAME, HF_DEVICE, HF_TOKEN))