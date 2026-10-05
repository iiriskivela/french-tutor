import io
import json
import os
from gtts import gTTS
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.set_page_config(page_title="Mon Tuteur Français", page_icon="🇫🇷", layout="centered")

# --- Custom CSS to dock the audio widget permanently to the bottom ---
st.markdown(
    """
    <style>
    /* Give the chat container bottom padding so the last message isn't hidden behind the fixed mic */
    .main .block-container {
        padding-bottom: 160px;
    }

    /* Target the audio input container and pin it above the chat input */
    [data-testid="stAudioInput"] {
        position: fixed;
        bottom: 80px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 700px;
        z-index: 999;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 6px 16px;
        border-radius: 14px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.06);
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        [data-testid="stAudioInput"] {
            background: rgba(14, 17, 23, 0.95);
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🇫🇷 Mon Ami & Tuteur Français")

# Groq API configuration
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY in .env or Streamlit secrets.")
    st.stop()

client = Groq(api_key=api_key)
CHAT_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"

SYSTEM_PROMPT = """
You are a dual-role French language tutor.
For every message sent by the user, respond ONLY with a valid JSON object matching this schema:
{
  "reply": "A friendly, conversational response written completely in natural French, continuing the conversation like a French friend.",
  "translation": "An accurate, natural English translation of your reply.",
  "corrections": [
    {
      "original": "The specific phrase the user used that had an issue",
      "better": "The corrected, natural native phrasing",
      "explanation": "Clear, concise reason why (grammar rule, unnatural phrasing, false friend, or gender agreement)"
    }
  ]
}

Guidelines:
- If the user's French has no errors and is natural, keep "corrections" as an empty list [].
- Never sound robotic in the "reply" — chat like a genuine peer.
- The output MUST be strictly valid JSON without any markdown formatting or code blocks.
"""

def generate_french_audio(text):
    """Generates an MP3 byte buffer of French speech."""
    try:
        tts = gTTS(text=text, lang="fr", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

if "messages" not in st.session_state:
    initial_text = "Salut ! Comment ça va aujourd'hui ? Tu as fait quoi de beau ?"
    st.session_state.messages = [
        {
            "role": "assistant",
            "reply": initial_text,
            "translation": "Hi! How are you doing today? Did you do anything fun?",
            "audio": generate_french_audio(initial_text),
            "corrections": [],
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg["reply"])

            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

            if msg.get("translation"):
                with st.expander("🇬🇧 Traduction en anglais"):
                    st.write(msg["translation"])

            if msg.get("corrections"):
                with st.expander("💡 Corrections & Explications"):
                    for c in msg["corrections"]:
                        st.markdown(f"- **Tu as dit :** *{c['original']}*")
                        st.markdown(f"- **Mieux vaut dire :** **{c['better']}**")
                        st.markdown(f"- *{c['explanation']}*\n")

# --- Persistent Bottom Audio + Text Input ---
audio_file = st.audio_input("🎙️ Enregistrer un message oral", label_visibility="collapsed")
typed_prompt = st.chat_input("Écris en français ici...")

# Handle audio transcription if recorded
spoken_prompt = None
if audio_file:
    audio_bytes = audio_file.read()
    if st.session_state.get("last_audio_bytes") != audio_bytes:
        st.session_state["last_audio_bytes"] = audio_bytes
        with st.spinner("Transcription de ta voix..."):
            try:
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", audio_bytes),
                    model=STT_MODEL,
                    language="fr",
                    prompt="Conversation en français courant.",
                )
                spoken_prompt = transcription.text
            except Exception as e:
                st.error(f"Erreur audio : {e}")

prompt = spoken_prompt if spoken_prompt else typed_prompt

# Send prompt to tutor
if prompt:
    if (
        not st.session_state.messages
        or st.session_state.messages[-1].get("content") != prompt
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                llm_history.append({"role": "user", "content": m["content"]})
            else:
                llm_history.append({"role": "assistant", "content": m["reply"]})
        llm_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("En train de réfléchir..."):
                try:
                    response = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=llm_history,
                        response_format={"type": "json_object"},
                        temperature=0.7,
                    )
                    data = json.loads(response.choices[0].message.content)

                    audio_stream = generate_french_audio(data["reply"])

                    st.markdown(data["reply"])
                    if audio_stream:
                        st.audio(audio_stream, format="audio/mp3")

                    if data.get("translation"):
                        with st.expander("🇬🇧 Traduction en anglais"):
                            st.write(data["translation"])

                    if data.get("corrections"):
                        with st.expander("💡 Corrections & Explications"):
                            for c in data["corrections"]:
                                st.markdown(f"- **Tu as dit :** *{c['original']}*")
                                st.markdown(
                                    f"- **Mieux vaut dire :** **{c['better']}**"
                                )
                                st.markdown(f"- *{c['explanation']}*\n")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "reply": data["reply"],
                            "translation": data.get("translation", ""),
                            "audio": audio_stream,
                            "corrections": data.get("corrections", []),
                        }
                    )
                except Exception as e:
                    st.error(f"Erreur API : {e}")