from src.llm.openai_llm import gpt_respond
from src.llm.hf_llm import hf_respond
from config import OPENAI_API, OPENAI_MODEL, HF_DEVICE, HF_MODEL_NAME, HF_TOKEN

# print(gpt_respond("Most exepensive house in Canada", OPENAI_API, OPENAI_MODEL))

print("\n##############\n")

print(hf_respond("Most expensive house in Canada", HF_MODEL_NAME, HF_DEVICE, HF_TOKEN))