from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_pipeline = None
conversation_history = []

def hf_respond(user_input: str, model_name: str, device: str = "auto", hf_token: str = None) -> str:
    global model_pipeline, conversation_history
    if hf_token:
        login(token=hf_token)
    if model_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype="auto", token=hf_token
        )
        model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True)
    conversation_history.append({"role": "user", "content": user_input})
    history = conversation_history[-10:]
    prompt = ""
    for msg in history:
        if msg["role"] == "user":
            prompt += "User: " + msg["content"] + "\n"
        else:
            prompt += "Assistant: " + msg["content"] + "\n"
    prompt += "Assistant:"
    outputs = model_pipeline(prompt, max_new_tokens=64, temperature=0.3, return_full_text=False)
    raw = outputs[0]["generated_text"]
    if raw.startswith(prompt):
        raw = raw[len(prompt):]
    for delim in ["\nUser:", "\nAssistant:"]:
        if delim in raw:
            raw = raw.split(delim)[0]
    response = raw.strip()
    conversation_history.append({"role": "assistant", "content": response})
    return response
