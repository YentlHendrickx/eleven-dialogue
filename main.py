import os

import dotenv
import requests
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

KOBOLD_URL = "http://localhost:5001/v1/chat/completions"
VOICE_ID = "8VJBrRk4O7tFoB5kqxa6"
SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are read aloud, so keep them "
    "short and conversational, and don't use markdown, lists or emojis."
)

def get_api_key() -> str:
    dotenv.load_dotenv()
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (check your .env)")
    return key


def get_llm_response(messages: list[dict], max_tokens: int = 300) -> str:
    """Send the conversation so far to KoboldCPP (OpenAI-compatible API)."""
    resp = requests.post(
        KOBOLD_URL,
        json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.7},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def speak(client: ElevenLabs, text: str, voice_id: str = VOICE_ID) -> None:
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)


def main():
    global VOICE_ID
    client = ElevenLabs(api_key=get_api_key())
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Type a message (empty line, 'quit' or Ctrl+C to exit).")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        if user_input.lower().split(' ')[0] == "/voice_id":
            new_voice_id = user_input.split(' ')[1] if len(user_input.split(' ')) > 1 else None
            if new_voice_id:
                VOICE_ID = new_voice_id
                print(f"Voice ID changed to: {VOICE_ID}")
            else:
                print("Please provide a voice ID after /voice_id command.")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            reply = get_llm_response(messages)
        except requests.RequestException as e:
            print(f"[KoboldCPP error: {e}]")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"AI: {reply}")
        speak(client, reply, voice_id=VOICE_ID)


if __name__ == "__main__":
    main()
