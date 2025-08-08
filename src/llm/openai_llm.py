from openai import OpenAI

conversation_history = [
    {
        'role': 'system',
        'content': "You are a voice-enabled research assistant designed to help users quickly find, summarize, and explain information from reliable sources. Prioritize accuracy and clarity. Respond in plain, english text ONLY. Do not use asteriks or anything that sounds weird in a TTS."
    }
]

def gpt_respond(prompt: str, api_key: str, model: str):
    global conversation_history

    client = OpenAI(api_key = api_key)

    print("Thinking...")

    try:
        conversation_history.append({'role': 'user', 'content': prompt})

        messages_to_send = [conversation_history[0]] + conversation_history[-10:]

        response = client.chat.completions.create(
            model = model,
            messages = messages_to_send,
            temperature = 0.3
        )

        assistant_reply = response.choices[0].message.content.strip()

        conversation_history.append({'role': 'assistant', 'content': assistant_reply})

        return assistant_reply

    except Exception as e:
        print(f'Error: {e}')
        return 'Could not generate a proper response'
