import streamlit as st
from openai import OpenAI
import tempfile

# ----------------
# ⚠️ REPLACE WITH YOUR OWN API KEY
# ----------------
API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

# ------------------------------------------
# HARDCODED CREDENTIALS
# ------------------------------------------
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"  # Change to your own

# ------------------------------------------
# PROMPTS
# ------------------------------------------
whisper_prompt = """
ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven,
DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.
"""

system_prompt = """
You are a helpful assistant tasked with assigning speakers to a transcribed conversation.
Ensure that the transcription maintains the correct order of dialogue while attempting to
separate different speakers. Also, correct spelling errors based on these product names:
ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven,
DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.

Keep dialogue formatting clean, and infer speaker names when necessary.
"""

summary_prompt = """
You are a helpful assistant. Please create a concise summary of the conversation.
The summary should capture key points, main topics, and important details without
revealing personal or sensitive information. Use bullet points or short paragraphs
for clarity.
"""

# ------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------
st.set_page_config(page_title="summary", layout="centered")
st.title("🎙️ Speech to Text")

# ------------------------------------------
# INITIALIZE SESSION STATE
# ------------------------------------------
if "speaker_assigned_transcript" not in st.session_state:
    st.session_state["speaker_assigned_transcript"] = ""

if "summary_text" not in st.session_state:
    st.session_state["summary_text"] = ""

if "has_summarized" not in st.session_state:
    st.session_state["has_summarized"] = False

# Track login status
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file with Whisper."""
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt=whisper_prompt,
                response_format="text"
            )
        return transcription.strip()
    except Exception as e:
        return f"⚠️ Error during transcription: {str(e)}"


def process_with_gpt4omini(system: str, user_input: str) -> str:
    """Send system + user messages to GPT-4o-mini and return the response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error processing text: {str(e)}"


# ------------------------------------------
# LOGIN/LOGOUT LOGIC
# ------------------------------------------
if not st.session_state["logged_in"]:
    # Show login form if not logged in
    st.subheader("🔒 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state["logged_in"] = True
            st.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")
else:
    # If logged in, show logout button and the main app
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["has_summarized"] = False
        st.session_state["summary_text"] = ""
        st.session_state["speaker_assigned_transcript"] = ""
        st.rerun()

    st.write("---")  # Visual separator

    # ------------------------------------------
    # FILE UPLOADER
    # ------------------------------------------
    uploaded_files = st.file_uploader(
        "📤 Upload audio files (mp3, wav, m4a)",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=True
    )

    # ------------------------------------------
    # MAIN LOGIC
    # ------------------------------------------
    if uploaded_files:
        # If there are uploaded files and we haven't summarized yet in this session,
        # let's do the full pipeline now.
        if not st.session_state["has_summarized"]:
            st.info("🔄 Processing and transcribing...")

            # Collect transcriptions from all uploaded files
            all_transcriptions = ""
            for uploaded_file in uploaded_files:
                st.write(f"### Processing `{uploaded_file.name}`")

                # Write to a temporary file (auto-delete after the with-block)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_file.flush()  # make sure data is written to disk

                    # Transcribe with Whisper
                    with st.spinner(f"🎙️ Transcribing `{uploaded_file.name}`..."):
                        raw_text = transcribe_audio(temp_file.name)

                # Show partial transcription
                st.text_area(
                    f"📝 Transcription for `{uploaded_file.name}`:",
                    raw_text,
                    height=100
                )

                # Append to our combined transcript
                all_transcriptions += (
                    f"\n\n--- Transcription from `{uploaded_file.name}` ---\n\n" + raw_text
                )

            # Once we have the full transcript, run speaker assignment & spelling correction
            with st.spinner("🔍 Assigning speakers and correcting spelling..."):
                final_transcript = process_with_gpt4omini(system_prompt, all_transcriptions)

            # Store in session state
            st.session_state["speaker_assigned_transcript"] = final_transcript
            st.success("✅ Speaker assignment completed!")

            # Now create ONE-TIME summary
            with st.spinner("📄 Creating summary..."):
                summary_text = process_with_gpt4omini(summary_prompt, final_transcript)

            st.session_state["summary_text"] = summary_text
            st.session_state["has_summarized"] = True  # We won't do this again in this session

        # Now display the results from session state
        st.subheader("🗣️ Speaker-Assigned Transcription")
        st.markdown(st.session_state["speaker_assigned_transcript"])

        st.subheader("📄 Summary (Generated Once)")
        st.text_area(
            "📝 Summary:",
            st.session_state["summary_text"],
            height=200
        )

    else:
        st.info("📤 Please upload one or more audio files to begin.")

    # ------------------------------------------
    # QUESTION-ANSWERING (Uses One-Time Summary)
    # ------------------------------------------
    st.subheader("🤖 Ask a Question")

    question = st.text_input("📝 Enter your question here:")
    if question:
        if not st.session_state["summary_text"]:
            st.warning("No summary available. Please upload files and wait for the summary.")
        else:
            with st.spinner("🤖 Thinking..."):
                qna_prompt = (
                    "Below is a summary of a conversation:\n\n"
                    f"{st.session_state['summary_text']}\n\n"
                    "Answer the question ONLY from this summary. If you are unsure, say you don't have enough information.\n\n"
                    f"Question: {question}"
                )
                answer = process_with_gpt4omini(
                    "You are a helpful assistant who only uses the provided summary as context.",
                    qna_prompt
                )
            st.write("💡 **Answer:**", answer)
