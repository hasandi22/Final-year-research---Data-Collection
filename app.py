import os
import io
import uuid
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

#from utils import scroll_to_top

# Define survey flow
steps = ["consent", "demographics", "baseline", "session_emp", "session_neu", "open", "review"]

# ElevenLabs
from elevenlabs.client import ElevenLabs

# Hugging Face Hub
from huggingface_hub import HfApi, hf_hub_download, HfFolder

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")        
HF_DATASET_PATH = os.getenv("HF_DATASET_PATH", "responses.csv")  

# Basic validations
if not ELEVENLABS_API_KEY:
    st.error("Missing ELEVENLABS_API_KEY in .env")
if not HF_TOKEN:
    st.error("Missing HF_TOKEN in .env")
if not HF_DATASET_REPO:
    st.error("Missing HF_DATASET_REPO in .env")

# Init clients
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
hf_api = HfApi()
HfFolder.save_token(HF_TOKEN)


st.set_page_config(page_title="Empathetic vs. Neutral AI Voice Study", page_icon="🎙", layout="centered")


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def play_voice(text: str, voice_name: str):
    """Generate & play audio from ElevenLabs dynamically."""
    try:
        voices_response = client.voices.get_all()
        voice_map = {v.name: v for v in voices_response.voices}

        if voice_name not in voice_map:
            st.error(f"Voice '{voice_name}' not found. Choose one from the dropdown.")
            return

        # Convert text to speech
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_map[voice_name].voice_id,
            model_id="eleven_flash_v2_5",
            text=text,
            output_format="mp3_22050_32"
        )

        # Collect chunks into one file
        audio_bytes = b"".join(audio_generator)

        st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")

    except Exception as e:
        st.error(f"Voice generation failed: {e}")

def load_existing_hf_csv(repo_id: str, path_in_repo: str) -> pd.DataFrame:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            token=HF_TOKEN
        )
        return pd.read_csv(local_path)
    except Exception:
        return pd.DataFrame(columns=["participant_id"])

def upload_csv_to_hf(df: pd.DataFrame, repo_id: str, path_in_repo: str):
    tmp_path = "responses_tmp.csv"
    df.to_csv(tmp_path, index=False)
    hf_api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )

def init_state():
    defaults = {
        "consented": False,
        "step": "consent",
        "participant_id": str(uuid.uuid4()),
        "start_ts": datetime.utcnow().isoformat(),
        # Demographics
        "age": None,
        "gender": None,
        "gender_other": "",
        "education": None,
        "voice_exp": None,
        "used_assistants": None,
        "tech_comfort": None,
        # GAD-7
        "gad": {f"q{i}": None for i in range(1, 8)},
        "gad_impact": None,
        # PANAS
        "panas": {f"q{i}": None for i in range(1, 11)},
        "single_mood": None,
        # Empathetic
        "emp": {f"q{i}": None for i in range(1, 9)},
        "emp_state_anxiety": None,
        "emp_post": {f"q{i}": None for i in range(1, 8)},
        # Neutral
        "neu": {f"q{i}": None for i in range(1, 9)},
        "neu_state_anxiety": None,
        "neu_post": {f"q{i}": None for i in range(1, 8)},
        # Open-ended
        "open_emp": "",
        "open_neu": "",
        "open_compare": "",
        "open_pref": "",
        "open_empathy": "",
        "open_trust": "",
        "open_triggers": "",
        "open_improve": "",
        "open_more_1": "",
        "open_more_2": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def section_header(text):
    st.markdown(f"### {text}")


init_state()
def scroll_to_top():
    if st.session_state.get("step_changed", False):
        st.markdown(
            "<script>window.scrollTo({top: 0, behavior: 'smooth'});</script>",
            unsafe_allow_html=True
        )
        st.session_state["step_changed"] = False

# Call this once, globally, before sections
scroll_to_top()

def show_progress():
    current_step = st.session_state.get("step", "consent")
    current_index = steps.index(current_step)
    progress = (current_index + 1) / len(steps)
    st.progress(progress)
    st.write(f"Step {current_index + 1} of {len(steps)}")

show_progress() # Show progress bar

# Call this immediately after session_state["step"] changes
def navigation_buttons(prev_step=None, next_step=None, prev_label="⬅ Back", next_label="Continue ➡"):
    cols = st.columns([1,1])
    with cols[0]:
        if prev_step and st.button(prev_label, key=f"back_{prev_step}"):
            st.session_state["step"] = prev_step
            st.session_state["step_changed"] = True
            st.rerun()
    with cols[1]:
        if next_step and st.button(next_label, key=f"next_{next_step}"):
            st.session_state["step"] = next_step
            st.session_state["step_changed"] = True
            st.rerun()




# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = "consent"
    

# -----------------------------
# CONSENT
# -----------------------------
if st.session_state["step"] == "consent":
    st.title("Empathetic vs. Neutral AI Voice Study")
    st.subheader("Informed Consent")
    st.write("""You are invited to participate in a study on how different AI voices affect emotional well-being. You will listen to two kinds of AI voices (one warm/empathetic and one neutral/robotic) and answer some questions.""")
    st.write("""Your participation is completely voluntary. You may skip any question or stop the study at any time without penalty.""")
    st.write("""The study takes about 15–20 minutes. You will first answer some questions about 
your background and current mood/anxiety. Then you will listen to AI voice 
recordings (one empathetic, one neutral) and respond to questions during and 
after each.
""")
    st.write("""Risks/Benefits: There are minimal risks. You may feel some anxiety recalling feelings, but no 
serious risks are expected. You will contribute to understanding how voice 
interactions can affect emotional health. """)
    st.write("""All your answers are confidential and anonymous. We will use the data only for 
research purposes. No personal identifiers (names, etc.) will be linked to your 
responses.""")
    st.write("""By continuing the survey, you acknowledge that you understand the information 
above and agree to participate. """)
    consent = st.checkbox("I agree to participate.")
    if st.button("Continue ➡"):
        if consent:
            st.session_state["consented"] = True
            st.session_state["step"] = "demographics"
            st.rerun()
        else:
            st.warning("You must agree to continue.")


# -----------------------------
# DEMOGRAPHICS
# -----------------------------

if st.session_state["step"] == "demographics":
    st.header("Demographic Information")
    st.session_state["age"] = st.number_input("Q1. Enter your age (years)", min_value=18, max_value=120, step=1)
    gender_choice = st.selectbox("Q2. Your gender",
                                 ["Female", "Male", "Non-binary/Other (specify)", "Prefer not to say"])
    st.session_state["gender"] = gender_choice
    if gender_choice == "Non-binary/Other (specify)":
        st.session_state["gender_other"] = st.text_input("Please specify:")
    st.session_state["education"] = st.selectbox(
        "Q3. Select your highest education level",
        ["High school or less", "Some college/Associate’s", "Bachelor’s degree", "Postgraduate degree"]
    )
    st.session_state["voice_exp"] = st.radio("Q4. Do you have any voice technology experience?", ["Yes", "No"])
    st.session_state["used_assistants"] = st.radio("Q5. Have you used voice assistants (e.g. Siri, Alexa)  before?", ["Yes", "No"])
    st.session_state["tech_comfort"] = st.radio("Q6.How comfortable are you with using technology (e.g., smartphones, computers, voice assistants)? ", ["Not at all", "Slightly", "Moderately", "Very", "Extremely"])


    navigation_buttons(prev_step="consent", next_step="baseline")


# -----------------------------
# Baseline: GAD-7 + PANAS + Mood
# -----------------------------
if st.session_state["step"] == "baseline":
    st.header("Baseline Mental Health and Mood")
    section_header("A. Anxiety – GAD-7 (Generalized Anxiety Disorder Scale)")
    st.write("""
The GAD-7 is a brief, standardized questionnaire used by clinicians and researchers 
to measure symptoms of generalized anxiety. It asks about common feelings and 
behaviors related to anxiety over the past two weeks. Your answers will help us 
understand your baseline level of anxiety before the voice sessions.
""")
    st.write("""Q7.Over the past 2 weeks, how often have you been bothered by the following problems? """)
    st.write("""Select an option for each question""")
    st.write("""Scale: 1 = Not at all 2 = Several days 3 = More than half the days 4 = Nearly every day .""")
    gad_items = [
        "Q7.1 Feeling nervous, anxious, or on edge.",
        "Q7.2 Not being able to stop or control worrying.",
        "Q7.3 Worrying too much about different things.",
        "Q7.4 Trouble relaxing.",
        "Q7.5 Being so restless that it is hard to sit still.",
        "Q7.6 Becoming easily annoyed or irritable.",
        "Q7.7 Feeling afraid as if something awful might happen."
    ]
    gad_scale = [1, 2, 3, 4]
    for i, label in enumerate(gad_items, start=1):
        st.session_state["gad"][f"q{i}"] = st.radio(label, gad_scale, horizontal=True)
    st.session_state["gad_impact"] = st.radio(" Q8. If you checked any problems above, how difficult have these made it for you to do your work, take care of things at home, or get along with other people??", ["Not difficult", "Somewhat", "Very", "Extremely"])

    section_header("B. Current Mood – PANAS - Positive and Negative Affect Schedule")
    st.write("""
The PANAS is a short questionnaire that measures positive and negative emotions. 
It helps us understand your current mood by asking how strongly you feel 
different emotions right now. This provides a snapshot of your emotional state 
before the voice sessions.
""")
    st.write("""Q9.Right now, to what extent do you feel each of the following emotions?""")
    st.write("""Select an option for each question""") 
    st.write("""Scale: 1 = Very slightly or not at all 2 = A little 3 = Moderately 4 = Quite a bit 5 = Extremely""")
    panas_items = ["Interested","Distressed","Excited","Upset","Strong","Guilty","Scared","Hostile(Aggressive)","Enthusiastic","Proud"]
    five_scale = [1,2,3,4,5]
    # Add question numbers to each item
    for i, label in enumerate(panas_items, start=1):
        st.session_state["panas"][f"q{i}"] = st.radio(f"Q9.{i} {label}", five_scale, horizontal=True)

    st.write("""Single-Item Mood Rating """)
    st.session_state["single_mood"] = st.radio("Q10.Overall, right now I feel… (1=very negative, 5=very positive):", [1,2,3,4,5], horizontal=True)

    navigation_buttons(prev_step="demographics", next_step="session_emp")

# Likert scale options
five_scale = ["1 = Strongly Disagree", "2", "3", "4", "5 = Strongly Agree"]

# Define questions
empathetic_questions = {
    "Q11": "I felt the voice was warm and caring.",
    "Q12": "The voice seemed to understand or respond to my feelings.",
    "Q13": "I felt comfortable listening to this voice.",
    "Q14": "The voice spoke in a calm, soothing tone.",
    "Q15": "I would trust this voice to give helpful advice.",
    "Q16": "The voice helped me feel supported.",
    "Q17": "The pace (speed) of the voice’s speech was comfortable.",
    "Q18": "I found it easy to pay attention to this voice."
}

neutral_questions = {
    "Q20": "The voice sounded neutral or robotic (monotone).",
    "Q21": "I felt the voice gave factual, impersonal responses.",
    "Q22": "I felt comfortable listening to this voice.",
    "Q23": "I would trust this voice to give accurate information.",
    "Q24": "The voice’s tone seemed emotionless.",
    "Q25": "The pace of the voice’s speech was comfortable.",
    "Q26": "I found it easy to pay attention to this voice.",
    "Q27": "The voice delivered the information clearly and understandably."
}

default_voice_metadata = {
    "Rachel": {"gender": "Female", "accent": "American", "description": "Casual, matter-of-fact, personable"},
    "Clyde": {"gender": "Male", "accent": "American", "description": "Intense, great for characters"},
    "Roger": {"gender": "Male", "accent": "American", "description": "Classy, easy-going"},
    "Sarah": {"gender": "Female", "accent": "American", "description": "Professional, confident, warm"},
    "Laura": {"gender": "Female", "accent": "American", "description": "Sassy, sunny enthusiasm, quirky"},
    "Thomas": {"gender": "Male", "accent": "American", "description": "Meditative, soft, subdued"},
    "Charlie": {"gender": "Male", "accent": "Australian", "description": "Hyped, confident, energetic"},
    "George": {"gender": "Male", "accent": "British", "description": "Mature, warm resonance"},
    "Callum": {"gender": "Male", "accent": "Neutral", "description": "Gravelly, unsettling edge"},
    "River": {"gender": "Neutral", "accent": "American", "description": "Calm, relaxed, neutral"},
    "Harry": {"gender": "Male", "accent": "American", "description": "Rough, animated warrior, young"},
    "Liam": {"gender": "Male", "accent": "American", "description": "Confident, energetic, warm, young"},
    "Alice": {"gender": "Female", "accent": "British", "description": "Clear, engaging, professional, friendly (e-learning suitable)"},
    "Matilda": {"gender": "Female", "accent": "American", "description": "Upbeat, professional, pleasing alto pitch, educational"},
    "Will": {"gender": "Male", "accent": "American", "description": "Chill, conversational, laid back, young"},
    "Jessica": {"gender": "Female", "accent": "American", "description": "Cute, young, playful, trendy"},
    "Eric": {"gender": "Male", "accent": "American", "description": "Classy, smooth tenor, middle-aged"},
    "Chris": {"gender": "Male", "accent": "American", "description": "Casual, natural, down-to-earth, middle-aged"},
    "Brian": {"gender": "Male", "accent": "American", "description": "Classy, resonant, comforting, middle-aged"},
    "Daniel": {"gender": "Male", "accent": "British", "description": "Formal, professional, broadcast/news, middle-aged"},
    "Lily": {"gender": "Female", "accent": "British", "description": "Confident, warm, velvety, narration, middle-aged"},
    "Bill": {"gender": "Male", "accent": "American", "description": "Crisp, friendly, comforting, old"},
}


# -----------------------------
# Empathetic Voice Session
# -----------------------------
if st.session_state["step"] == "session_emp":
    st.header("Empathetic Voice Session")
    st.write("""Instructions: For each voice session, please rate the following statements about that voice on a 5-point scale:""")

    # Fetch all ElevenLabs voices
    voices = client.voices.get_all().voices

    # Build labels with fallback to hardcoded metadata
    voice_labels = {}
    for v in voices:
        meta = default_voice_metadata.get(v.name, {})
        gender = v.labels.get("gender") or meta.get("gender", "Unknown")
        accent = v.labels.get("accent") or meta.get("accent", "Unknown")
        desc   = v.labels.get("description") or meta.get("description", "No description available")
        label  = f"{v.name} — {gender} | {accent} | {desc}"
        voice_labels[label] = v

    # Select empathetic voice
    emp_voice_label = st.selectbox(
        "Choose empathetic voice:",
        list(voice_labels.keys()),
        key="emp_voice_select"
    )
    emp_voice = voice_labels[emp_voice_label].name

    # Text area for empathetic script
    emp_script = st.text_area(
        "Empathetic script:",
        """Hi, I’m glad you’re here. I know life can feel overwhelming sometimes, 
and it’s completely okay to have moments of stress or worry. 
You’re not alone in feeling this way. Take a slow breath with me… inhale… and exhale. 
You’re doing your best, and that’s enough. Remember, even small steps forward matter. 
You deserve kindness, and I’m proud of you for taking this moment for yourself.""",
        key="emp_script_text"
    )

    # Play button
    if st.button("▶ Play Empathetic Voice", key="emp_play_btn"):
        play_voice(emp_script, emp_voice)

    st.subheader("AI Voice Interaction Questions (Empathetic Voice)")
    for i, (key, question) in enumerate(empathetic_questions.items(), start=11):
        st.radio(f"Q{i}. {question}", five_scale, key=key, horizontal=True)

    st.subheader("During-Interaction Anxiety (State Anxiety)")

    st.write("""Q19.After this empathetic voice session, please indicate how anxious you felt during the session by selecting a number from 1 to 5:""")

    st.session_state["emp_state_anxiety"] = st.radio(
        "",
        [1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} = {['Not at all anxious','Slightly anxious','Moderately anxious','Very anxious','Extremely anxious'][x-1]}"
    )


    navigation_buttons(prev_step="baseline", next_step="session_neu")


# -----------------------------
# Neutral Voice Session
# -----------------------------   
if st.session_state["step"] == "session_neu":
    st.header("Neutral / Robotic Voice Session")
    st.write("""Instructions: For each voice session, please rate the following statements about that voice on a 5-point scale:""")

    # Fetch all ElevenLabs voices
    voices = client.voices.get_all().voices

    # Build labels with fallback to hardcoded metadata
    voice_labels = {}
    for v in voices:
        meta = default_voice_metadata.get(v.name, {})
        gender = v.labels.get("gender") or meta.get("gender", "Unknown")
        accent = v.labels.get("accent") or meta.get("accent", "Unknown")
        desc   = v.labels.get("description") or meta.get("description", "No description available")
        label  = f"{v.name} — {gender} | {accent} | {desc}"
        voice_labels[label] = v

    # Select neutral voice
    neu_voice_label = st.selectbox(
        "Choose neutral voice:",
        list(voice_labels.keys()),
        key="neu_voice_select"
    )
    neu_voice = voice_labels[neu_voice_label].name

   # Text area for neutral script
    neu_script = st.text_area(
        "Neutral script:",
        """Hello, thank you for participating in this session. 
In a moment, you will be asked to reflect on your current feelings. 
This is simply a part of the study procedure. 
Please listen carefully and respond as instructed. 
There are no right or wrong answers. 
Your participation is valuable, and your responses will help us better understand voice interactions.""",
        key="neu_script_text"
    )

    # Play button
    if st.button("▶ Play Neutral Voice", key="neu_play_btn"):
        play_voice(neu_script, neu_voice)

    st.subheader("AI Voice Interaction Questions (Neutral Voice)")
    for i, (key, question) in enumerate(neutral_questions.items(), start=20):
        st.radio(f"Q{i}. {question}", five_scale, key=key, horizontal=True)

    st.subheader("During-Interaction Anxiety (State Anxiety)")

    st.write("""Q28.After this robotic voice session, please indicate how anxious you felt during the session by selecting a number from 1 to 5:""")

    st.session_state["neu_state_anxiety"] = st.radio(
        "",
        [1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} = {['Not at all anxious','Slightly anxious','Moderately anxious','Very anxious','Extremely anxious'][x-1]}"
    )

    navigation_buttons(prev_step="session_emp", next_step="open")



# -----------------------------
# Open-Ended Feedback
# -----------------------------
if st.session_state["step"] == "open":
    st.header("Open-Ended Qualitative Questions")
    st.write("**Feel free to write as much as you like; there are no right or wrong answers.**")
    st.write("**Empathetic Voice Experience**")
    st.session_state["open_emp"] = st.text_area("Q29.How did you feel during and after interacting with the empathetic AI voice? What kinds of emotions, thoughts, or reactions did it bring up for you? ")
    st.write("**Neutral Voice Experience**")
    st.session_state["open_neu"] = st.text_area("Q30.How did you feel during and after interacting with the neutral or robotic AI voice? What kinds of emotions, thoughts, or reactions did it bring up for you?")
    st.write("**Comparison of voices**")
    st.session_state["open_compare"] = st.text_area("Q31.What differences, if any, did you notice between the two voices in terms of how they made you feel? Which one made you feel more comfortable or anxious, and why? ")
    st.write("**Voice Preference**")
    st.session_state["open_pref"] = st.text_area("Q32.Which voice did you prefer overall? What specific features (tone, pace, warmth, etc.) did you like or dislike about each voice?")
    st.write("**Perceived Empathy and Understanding**")
    st.session_state["open_empathy"] = st.text_area("Q33.Did the empathetic voice make you feel understood or cared for in any way? If so, can you describe a moment or response that gave you that feeling? ")
    st.write("**Trust & Usefulness**")
    st.session_state["open_trust"] = st.text_area("Q34.Did you feel that either voice was trustworthy or helpful? Why or why not? In what ways did the voice help (or fail to help) you feel supported? ")
    st.write("**Triggers and Discomfort**")
    st.session_state["open_triggers"] = st.text_area("Q35.Was there anything in either voice interaction that made you feel uneasy, anxious, or emotionally uncomfortable? Please explain if so.")
    st.write("**Improvement Suggestions**")
    st.session_state["open_improve"] = st.text_area("Q36.If you could improve or change anything about the voices or how the interaction worked, what would you recommend to make it more helpful or emotionally supportive? ")
    st.write("**Additional Reflections**")
    st.session_state["open_more_1"] = st.text_area("Q37.Is there anything else you’d like to share about your experience in this study?")
    st.session_state["open_more_2"] = st.text_area("Q38.Any thoughts that haven’t been covered by the previous questions?")


    navigation_buttons(prev_step="session_neu", next_step="review", next_label="Review & Submit ➡")

# -----------------------------
# Review & Submit
# -----------------------------
if st.session_state["step"] == "review":
    st.header("Review & Submit")
    st.write("Click **Submit** to upload your responses")
    if st.button("Submit"):
        record = {
            "participant_id": st.session_state["participant_id"], 
            "start_ts_utc": st.session_state["start_ts"], 
            "submit_ts_utc": datetime.utcnow().isoformat(),
            "age": st.session_state["age"], "gender": st.session_state["gender"], 
            "gender_other": st.session_state.get("gender_other", ""), 
            "education": st.session_state["education"],
            "voice_exp": st.session_state["voice_exp"], "used_assistants": st.session_state["used_assistants"],
            "tech_comfort": st.session_state["tech_comfort"], 
            "single_mood": st.session_state["single_mood"]
        }
        for i in range(1,8): record[f"gad_q{i}"]=st.session_state["gad"][f"q{i}"]
        record["gad_impact"]=st.session_state["gad_impact"]
        for i in range(1,11): record[f"panas_q{i}"]=st.session_state["panas"][f"q{i}"]
        for i in range(1,9): record[f"emp_q{i}"]=st.session_state["emp"][f"q{i}"]
        record["emp_state_anxiety"]=st.session_state["emp_state_anxiety"]
        for i in range(1,8): record[f"emp_post_q{i}"]=st.session_state["emp_post"][f"q{i}"]
        for i in range(1,9): record[f"neu_q{i}"]=st.session_state["neu"][f"q{i}"]
        record["neu_state_anxiety"]=st.session_state["neu_state_anxiety"]
        for i in range(1,8): record[f"neu_post_q{i}"]=st.session_state["neu_post"][f"q{i}"]
        record.update({k: st.session_state[k] for k in ["open_emp","open_neu","open_compare","open_pref",
                                                         "open_empathy","open_trust","open_triggers",
                                                         "open_improve","open_more_1","open_more_2"]})
        try:
            existing_df = load_existing_hf_csv(HF_DATASET_REPO, HF_DATASET_PATH)
            updated_df = pd.concat([existing_df,pd.DataFrame([record])],ignore_index=True)
            upload_csv_to_hf(updated_df, HF_DATASET_REPO, HF_DATASET_PATH)
            st.success("Submitted successfully!")
            #st.info(f"Repo: {HF_DATASET_REPO} | File: {HF_DATASET_PATH}")
        except Exception as e:
            st.error(f"Upload failed: {e}")
