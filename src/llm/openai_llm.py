from openai import OpenAI

conversation_history = [
    {
        'role': 'system',
        'content': """You are a voice-enabled research assistant designed to help users quickly find, summarize, and explain information from reliable sources. 
        You respond in a conversational, engaging, and easy-to-understand manner suitable for spoken delivery. 
        When mentioning sources, do not use formal citations; instead, refer to them naturally in the conversation (e.g., 'according to WWF, pandas are dying!'). 
        Always prioritize accuracy, clarity, and reliability. When asked for complex information, break it into simple parts and explain them clearly. 
        If you do not know the answer, be upfront about it and suggest next steps. Avoid speculation and keep responses concise while still informative. Respond in English."""
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
