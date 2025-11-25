import streamlit as st
import torch
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from io import BytesIO
from reportlab.pdfgen import canvas
from datetime import datetime
import PyPDF2
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from textwrap import wrap  
import matplotlib.pyplot as plt
import os


## ---------------------- SESSION STATE -------------------------
if "text" not in st.session_state:
    st.session_state["text"] = ""

# Initialize analysis state to avoid KeyError
if "analysis" not in st.session_state:
    st.session_state["analysis"] = {
        "fake_label": "",
        "ai_label": "",
        "fake_prob": 0,
        "real_prob": 0,
        "human_prob": 0,
        "ai_prob": 0
    }

# ---------------------- MODEL PATHS ----------------------

# Assuming this file is at src/app/streamlit_app.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

FAKE_MODEL_PATH = os.path.join(REPO_ROOT, "model", "final_model")
AI_MODEL_PATH = os.path.join(REPO_ROOT, "model", "ai_detector_model")

# Normalize paths
FAKE_MODEL_PATH = os.path.normpath(FAKE_MODEL_PATH)
AI_MODEL_PATH = os.path.normpath(AI_MODEL_PATH)


# ---------------------- MODEL LOADING -------------------------


@st.cache_resource
def load_models():
    print("Loading Fake model...")
    tokenizer_fake = AutoTokenizer.from_pretrained(FAKE_MODEL_PATH)
    model_fake = AutoModelForSequenceClassification.from_pretrained(FAKE_MODEL_PATH)
    print("Fake model loaded!")

    print("Loading AI model...")
    tokenizer_ai = AutoTokenizer.from_pretrained(AI_MODEL_PATH)
    model_ai = AutoModelForSequenceClassification.from_pretrained(AI_MODEL_PATH)
    print("AI model loaded!")

    return tokenizer_fake, model_fake, tokenizer_ai, model_ai

tokenizer_fake, model_fake, tokenizer_ai, model_ai = load_models()

# ---------------------- PIPELINE CREATION -------------------------
@st.cache_resource
def create_pipelines():
    classifier_fake = pipeline(
        "text-classification", model=model_fake, tokenizer=tokenizer_fake, return_all_scores=True
    )
    classifier_ai = pipeline(
        "text-classification", model=model_ai, tokenizer=tokenizer_ai, return_all_scores=True
    )
    return classifier_fake, classifier_ai

classifier_fake, classifier_ai = create_pipelines()

# ---------------------- CLASSIFY FUNCTIONS -------------------------
def classify_fake(text, chunk_size=512):
    inputs = tokenizer_fake(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]
    
    scores_list = []
    progress = st.progress(0)  # progress bar
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_inputs = {"input_ids": chunk.unsqueeze(0)}
        if "attention_mask" in inputs:
            chunk_inputs["attention_mask"] = inputs["attention_mask"][0][i*chunk_size : i*chunk_size + len(chunk)].unsqueeze(0)
        outputs = model_fake(**chunk_inputs)
        scores = torch.softmax(outputs.logits, dim=-1)
        scores_list.append(scores[0].detach().numpy())
        
        progress.progress(int((i+1)/total_chunks*100))  # update progress
    
    avg_scores = torch.tensor(scores_list).mean(dim=0)
    return [{"label": "REAL", "score": avg_scores[0].item()}, {"label": "FAKE", "score": avg_scores[1].item()}]

def classify_ai(text, chunk_size=512):
    inputs = tokenizer_ai(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]
    
    scores_list = []
    progress = st.progress(0)
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_inputs = {"input_ids": chunk.unsqueeze(0)}
        if "attention_mask" in inputs:
            chunk_inputs["attention_mask"] = inputs["attention_mask"][0][i*chunk_size : i*chunk_size + len(chunk)].unsqueeze(0)
        outputs = model_ai(**chunk_inputs)
        scores = torch.softmax(outputs.logits, dim=-1)
        scores_list.append(scores[0].detach().numpy())
        
        progress.progress(int((i+1)/total_chunks*100))
    
    avg_scores = torch.tensor(scores_list).mean(dim=0)
    return [{"label": "HUMAN", "score": avg_scores[0].item()}, {"label": "AI", "score": avg_scores[1].item()}]

# ---------------------- UI CUSTOM STYLING -------------------------
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background-color: #121212;
        color: #BB86FC;
    }

    /* Sidebar labels and input labels */
    label, .stTextInput>label, .stTextArea>label, .stRadio>div>label>div {
        color: #BB86FC !important;
    }

    /* Text area for typed text or uploaded content */
    .stTextArea>div>div>textarea,
    .stTextArea>div>div>div>textarea {
        background-color: #1E1E1E !important;  /* dark background */
        color: #BB86FC !important;             /* purple text */
        border: 1px solid #03DAC6 !important;  /* teal border */
        font-size: 14px;
        height: auto;
        min-height: 150px;
        max-height: 400px;
    }

    /* Placeholder text */
    .stTextArea>div>div>textarea::placeholder {
        color: #BB86FC !important;
    }

    /* File uploader text */
    .stFileUploader>div>div>label, .stFileUploader>div>div>p {
        color: #03DAC6 !important;
    }

    /* Buttons */
    button {
        background-color: #BB86FC !important;
        color: #121212 !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Example of existing UI code below
st.title("🧠 Fake News & AI Text Detection System")

input_mode = st.radio(
    "Select Input Method:",
    ["📝 Type Text", "🎤 Voice Input", "📂 Upload File"]
)


# ---------------------- Handle Inputs -------------------------
# 1️⃣ Manual typing
if input_mode == "📝 Type Text":
    st.session_state["text"] = st.text_area("Enter text to analyze:", st.session_state["text"], height=150)

# 2️⃣ Voice Input
# 2️⃣ Voice Input
elif input_mode == "🎤 Voice Input":
    if st.button("🎙 Start Recording"):
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as mic:
                st.write("Listening... Speak now.")
                audio = recognizer.listen(mic)
                text = recognizer.recognize_google(audio)
                st.session_state["text"] = text  # replace existing text
                st.success("Captured voice input!")
        except Exception as e:
            st.error(f"Voice Recognition Error: {e}")

# 3️⃣ File Upload
elif input_mode == "📂 Upload File":
    file = st.file_uploader("Upload TXT / PDF / CSV / AUDIO (wav/mp3)", type=["txt", "pdf", "csv", "wav", "mp3"])
    if file:
        if file.type == "text/plain":
            st.session_state["text"] = file.read().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            extracted = " ".join(page.extract_text() for page in reader.pages)
            st.session_state["text"] = extracted
        elif "csv" in file.type:
            df = pd.read_csv(file)
            # Combine all rows and all columns into a single string
            combined_text = " ".join(df.astype(str).agg(' '.join, axis=1))
            st.session_state["text"] = combined_text
        elif "audio" in file.type:
            wav, sr_rate = librosa.load(file, sr=None)
            recognizer = sr.Recognizer()
            audio_file = sr.AudioData(wav.tobytes(), sr_rate, 2)
            extracted = recognizer.recognize_google(audio_file)
            st.session_state["text"] = extracted

# ---------------------- SHOW TEXT -------------------------
if st.session_state["text"]:
    st.text_area("📌 Text to Analyze:", st.session_state["text"], height=200)


# ---------------------- ANALYZE BUTTON -------------------------
if st.button("🔍 Analyze Text"):
    text = st.session_state["text"].strip()
    
    if not text:
        st.error("⚠ Please enter or upload text first.")
    else:
        # ----- SHORT TEXT CHECK -----
        if len(text.split()) < 5:  # adjust word threshold
            st.info(
                "ℹ️ Note: The input text is quite short. This system is optimized for analyzing longer content "
                "such as news articles, assignments, AI-generated text, or code snippets. "
                "Short inputs may result in less reliable predictions."
            )
            proceed_short = st.button("Analyze short text anyway")
            if not proceed_short:
                st.stop()  # pause analysis until user chooses to continue

        # ----- LARGE FILE WARNING -----
        if len(text) > 5000:
            st.info("⚠ Large file detected. Analysis may take a few minutes, please wait...")

        # ----- ANALYSIS -----
        with st.spinner("Analyzing text... ⏳ This may take a while for large files."):
            try:
                fake_out = classify_fake(text)
                ai_out   = classify_ai(text)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                fake_out = [{"label": "REAL", "score": 0}, {"label": "FAKE", "score": 0}]
                ai_out = [{"label": "HUMAN", "score": 0}, {"label": "AI", "score": 0}]

        # Convert scores to floats
        fake_out = [{"label": d["label"], "score": float(d["score"])} for d in fake_out]
        ai_out = [{"label": d["label"], "score": float(d["score"])} for d in ai_out]

        # Get labels and probabilities
        fake_label = max(fake_out, key=lambda x: x['score'])['label']
        ai_label = max(ai_out, key=lambda x: x['score'])['label']

        real_prob = fake_out[0]["score"] * 100
        fake_prob = fake_out[1]["score"] * 100
        human_prob = ai_out[0]["score"] * 100
        ai_prob = ai_out[1]["score"] * 100

        st.success("✔ Analysis Complete!")
        st.write(f"📰 Fake News Detection: **{fake_label}** ({fake_prob:.2f}%)")
        st.write(f"🤖 AI Text Detection: **{ai_label}** ({ai_prob:.2f}%)")

        # Store analysis in session state
        st.session_state["analysis"] = {
            "fake_label": fake_label,
            "ai_label": ai_label,
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "human_prob": human_prob,
            "ai_prob": ai_prob
        }


 # ---------------------- CHARTS -------------------------
a = st.session_state.get("analysis", None)
if a:
    # Replace NaN or None with 0
    real_prob = float(a.get("real_prob", 0) or 0)
    fake_prob = float(a.get("fake_prob", 0) or 0)
    human_prob = float(a.get("human_prob", 0) or 0)
    ai_prob = float(a.get("ai_prob", 0) or 0)

   # Bar chart: Fake vs Real (Purple vs Teal)
fig, ax = plt.subplots(facecolor="#121212")  # Dark background
ax.bar(
    ["Real", "Fake"],
    [real_prob, fake_prob],
    color=["#BB86FC", "#03DAC6"]  # Purple and teal
)

ax.set_ylabel("Probability %", color="white")
# Title: text letters in purple, emojis in system default color
ax.set_title("📰🧠 Fake News Confidence", color="#BB86FC", fontname="Segoe UI Emoji")

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')

st.pyplot(fig, facecolor="#121212")

# Pie chart: AI vs Human (Teal vs Purple)
fig2, ax2 = plt.subplots(facecolor="#121212")

# Prevent NaN by using default 50-50 if both probs are 0
if human_prob + ai_prob == 0:
    human_prob, ai_prob = 50, 50

ax2.pie(
    [human_prob, ai_prob],
    labels=["Human", "AI"],
    autopct="%1.1f%%",
    colors=["#03DAC6", "#BB86FC"],  # Teal and purple
    textprops={"color": "white", "fontsize": 12}
)

# Title with real-colored emojis
ax2.set_title("🤖👤 AI Text Detection Confidence", color="#BB86FC", fontname="Segoe UI Emoji")

st.pyplot(fig2, facecolor="#121212")


# ---------------------- CONFIDENCE METER -------------------------
st.write("### 📊 Confidence Meter")

if a:
    # Fake News Confidence
    st.write(f"📰🧠 Fake News Confidence: {fake_prob:.2f}% Fake | {real_prob:.2f}% Real")
    st.progress(int(fake_prob))  # shows percentage of Fake
    st.progress(int(real_prob))  # optional second bar for Real

    # AI Detection Confidence
    st.write(f"🤖👤 AI Detection Confidence: {ai_prob:.2f}% AI | {human_prob:.2f}% Human")
    st.progress(int(ai_prob))  # shows percentage of AI
    st.progress(int(human_prob))  # optional second bar for Human



# ---------------------- EXPORT PDF -------------------------
from textwrap import wrap  # make sure this is at the top of your file

if st.button("📄 Export Report"):
    if "analysis" not in st.session_state or not st.session_state["text"]:
        st.error("⚠ Run analysis before exporting.")
    else:
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer)

        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(50, 800, "Fake News & AI Detection Report")

        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, 780, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        a = st.session_state["analysis"]

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, 750, f"Fake News Result: {a['fake_label']}")
        pdf.drawString(50, 730, f"AI Detection Result: {a['ai_label']}")

        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, 700, f"Real Probability: {a['real_prob']:.2f}%")
        pdf.drawString(50, 680, f"Fake Probability: {a['fake_prob']:.2f}%")
        pdf.drawString(50, 660, f"Human Probability: {a['human_prob']:.2f}%")
        pdf.drawString(50, 640, f"AI-Generated Probability: {a['ai_prob']:.2f}%")

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, 610, "Analyzed Text:")

        pdf.setFont("Helvetica", 9)
        y = 590
        for paragraph in st.session_state["text"].split("\n"):
            wrapped_lines = wrap(paragraph, 90)  # wrap at 90 chars per line
            for line in wrapped_lines:
                if y < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 9)
                    y = 800
                pdf.drawString(50, y, line)
                y -= 12

        # Save PDF and enable download (outside the loop)
        pdf.save()
        buffer.seek(0)
        st.download_button("⬇ Download PDF", buffer, "FakeNews_Report.pdf", "application/pdf")
