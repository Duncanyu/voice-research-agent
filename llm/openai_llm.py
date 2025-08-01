from openai import OpenAI
from config import OPENAI_API, OPENAI_MODEL

def respond(prompt : str):
    client = OpenAI(api_key = OPENAI_API)
    
    try:
        response = client.chat.completions.create(
            model = OPENAI_MODEL,
            messages = [
                {'role': 'system', 'content': """You are a voice-enabled research assistant designed to help users quickly find, summarize, and explain information from reliable sources. 
                 You respond in a conversational, engaging, and easy-to-understand manner suitable for spoken delivery. 
                 When mentioning sources, do not use formal citations; instead, refer to them naturally in the conversation (e.g., 'according to WWF, pandas are dying!'). 
                 Always prioritize accuracy, clarity, and reliability. When asked for complex information, break it into simple parts and explain them clearly. 
                 If you do not know the answer, be upfront about it and suggest next steps. Avoid speculation and keep responses concise while still informative."""},
                {'role': 'user', 'content': prompt}
                ],
            temperature = 0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f'Error: {e}')
        return 'Could not generate a proper response'

print(respond("What is the most expensive house in the world?"))