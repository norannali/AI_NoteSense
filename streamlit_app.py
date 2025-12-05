import streamlit as st
from adaptive_engine.adaptive_engine import AdaptiveEngine
from core_models.extractor import extract_pdf
from core_models.pdf_summarizer import summarize_text
from providers.OpenRouterProvider import OpenRouterProvider

llm = OpenRouterProvider()

@st.cache_resource
def load_engine():
    return AdaptiveEngine()

engine = load_engine()

st.set_page_config(page_title="AI Adaptive Tutor", layout="wide")
st.title("🎓 AI Adaptive Tutor – Personalized Learning Engine")

user_id = st.text_input("👤 User ID", value="student_01")
level = st.selectbox("📚 Student Level", ["beginner", "intermediate", "advanced"])
engine.memory.set_user_level(user_id, level)

uploaded_pdf = st.file_uploader("📄 Upload Lecture PDF (optional)", type=["pdf"])
question = st.text_area("💬 Enter your question or text")

task = st.selectbox("✨ AI Task", ["Explain", "Summarize", "Explain Differently"])

if st.button("Generate Response"):

    lecture_summary = None

    if uploaded_pdf:
        with st.spinner("📄 Extracting PDF..."):
            pdf_text = extract_pdf(uploaded_pdf)

        with st.spinner("✨ Summarizing PDF..."):
            lecture_summary = summarize_text(pdf_text)

    # final input
    if lecture_summary:
        final_input = f"{question}\n\nRelated Lecture Summary:\n{lecture_summary}"
    else:
        final_input = question

    with st.spinner("🤖 AI thinking..."):

        # main output (llm)
        if task == "Summarize":
            result_text = summarize_text(final_input)
            mode = "summarize"

        elif task == "Explain":
            result_text = llm.explain(final_input, level)
            mode = "explain"

        elif task == "Explain Differently":
            result_text = llm.explain_differently(final_input)
            mode = "explain_differently"

        # metadata
        meta = engine.process(user_id, final_input, level, mode=mode)

    st.subheader("🧠 AI Response")
    st.write(result_text)

    st.subheader("📊 Metadata")
    st.json(meta["metadata"])
