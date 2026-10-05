import json
import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.set_page_config(page_title="Mon Tuteur Français", page_icon="🇫🇷")
st.title("🇫🇷 Mon Ami & Tuteur Français")

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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "reply": "Salut ! Comment ça va aujourd'hui ? Tu as fait quoi de beau ?",
            "translation": "Hi! How are you doing today? Did you do anything fun?",
            "corrections": [],
        }
    ]

# Native browser French voice player
def speak_french(text):
    clean_text = json.dumps(text)
    components.html(
        f"""
        <script>
        const utter = new SpeechSynthesisUtterance({clean_text});
        utter.lang = 'fr-FR';
        utter.rate = 0.95;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utter);
        </script>
        """,
        height=0,
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg["reply"])

            # English Translation Dropdown
            if msg.get("translation"):
                with st.expander("🇬🇧 Traduction en anglais"):
                    st.write(msg["translation"])

            # Corrections Dropdown
            if msg.get("corrections"):
                with st.expander("💡 Corrections & Explications"):
                    for c in msg["corrections"]:
                        st.markdown(f"- **Tu as dit :** *{c['original']}*")
                        st.markdown(f"- **Mieux vaut dire :** **{c['better']}**")
                        st.markdown(f"- *{c['explanation']}*\n")

# --- Voice Input in Sidebar ---
spoken_prompt = None
with st.sidebar:
    st.header("🎙️ Parle en français")
    st.caption("Enregistre ton message oral :")
    audio_file = st.audio_input("Microphone")
    if audio_file:
        with st.spinner("Transcription de ta voix..."):
            try:
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", audio_file.read()),
                    model=STT_MODEL,
                    language="fr",
                    prompt="Conversation en français courant.",
                )
                spoken_prompt = transcription.text
                st.success(f"Compris : « {spoken_prompt} »")
            except Exception as e:
                st.error(f"Erreur audio : {e}")

# Text Input at Bottom
typed_prompt = st.chat_input("Ou écris en français ici...")

# Prioritize voice if recorded, otherwise text
prompt = spoken_prompt if spoken_prompt else typed_prompt

# Process the message
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

                    # Show response and play audio
                    st.markdown(data["reply"])
                    speak_french(data["reply"])

                    # Show Translation dropdown
                    if data.get("translation"):
                        with st.expander("🇬🇧 Traduction en anglais"):
                            st.write(data["translation"])

                    # Show Corrections dropdown
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
                            "corrections": data.get("corrections", []),
                        }
                    )
                except Exception as e:
                    st.error(f"Erreur API : {e}")