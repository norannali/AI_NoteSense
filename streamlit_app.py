import streamlit as st
import re
from adaptive_engine.adaptive_engine import AdaptiveEngine
from core_models.extractor import extract_pdf
from providers.OpenRouterProvider import OpenRouterProvider
import graphviz

# --- Page Configuration ---
st.set_page_config(page_title="AI NoteSense", layout="wide")

# --- Load Core Engine & Models (Cached) ---
@st.cache_resource
def load_engine():
    return AdaptiveEngine()

engine = load_engine()
llm = OpenRouterProvider()

# --- Sidebar: Session Controls & User Settings ---
with st.sidebar:
    st.title("⚙️ Controls")
    
    # User Identification
    user_id = st.text_input("User ID", value="student_01")
    level = st.selectbox("Level", ["beginner", "intermediate", "advanced"])
    
    st.divider()
    
    # [NEW] Feature: AI Task Selection
    task_mode = st.selectbox(
        "🤖 AI Task Mode",
        ["General Chat", "Summarize Document", "Explain Concept", "Explain Differently", "Visual Outline"]
    )
    
    # Run Task Button
    run_task = st.button("🚀 Run Task Now", type="primary")
    
    st.divider()

    # Session History Control
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.session_state.topic = None
        st.rerun()

# --- Main Interface ---
st.title("🎓 AI NoteSense – Intelligent Study Assistant")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = None

# --- Feature 1: File Upload & Processing ---
uploaded_pdf = st.file_uploader("📄 Upload Lecture PDF (Optional)", type=["pdf"])
pdf_text = ""

if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_pdf(uploaded_pdf)
        
        # Feature 2: Auto Topic Detection
        if not st.session_state.topic:
            # We pass only the first 1000 chars to detect topic quickly
            detected_topic = llm.detect_topic(pdf_text[:1000])
            st.session_state.topic = detected_topic
            st.success(f"📌 Detected Topic: {detected_topic}")

# --- Feature 7: Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "visual" in msg:
            st.graphviz_chart(msg["visual"])

# --- Logic: Handle Input (Either from Chat OR Button) ---

# 1. Check Chat Input
chat_input = st.chat_input("Type your question or request here...")

# 2. Determine the Final Prompt
final_prompt = None

if chat_input:
    # User typed a question manually
    final_prompt = chat_input
elif run_task:
    # User clicked "Run Task Now"
    # Create a default prompt based on the selected mode
    if task_mode == "Summarize Document":
        final_prompt = "Please summarize the uploaded document based on the context provided."
    elif task_mode == "Visual Outline":
        final_prompt = "Create a visual concept map for this document."
    elif task_mode == "Explain Concept":
        final_prompt = "Explain the main concepts in this document."
    elif task_mode == "Explain Differently":
        final_prompt = "Explain the content in 3 different ways (Simple, Example, Technical)."
    else:
        final_prompt = "Hello AI, I'm ready to study."

# --- Processing The Request ---
if final_prompt:
    
    # 1. Append User Message to History
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    # 2. Prepare Context (Critical Step!)
    # We combine the User Question + The File Content
    context = pdf_text if pdf_text else ""
    full_input = f"{final_prompt}\n\nContext:\n{context[:4000]}" # Passed 4000 chars of context

    # 3. Generate Response
    with st.chat_message("assistant"):
        with st.spinner(f"AI is running task: {task_mode}..."):
            
            # --- Handling Visual Outline Task ---
            if task_mode == "Visual Outline" or "visual" in final_prompt.lower():
                response_text = "Here is the visual concept map for your request:"
                # Here we used full_input correctly
                dot_code = llm.generate_visual_outline(full_input)
                clean_dot = re.sub(r'```dot|```', '', dot_code).strip()
                
                try:
                    st.graphviz_chart(clean_dot)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text, 
                        "visual": clean_dot
                    })
                except Exception as e:
                    st.error("Could not render chart. Raw code provided.")
                    st.code(clean_dot)
            
            else:
                # --- Mapping UI Task to Engine Mode ---
                mode_map = {
                    "General Chat": "answer",
                    "Summarize Document": "summarize",
                    "Explain Concept": "explain",
                    "Explain Differently": "explain_differently"
                }
                
                selected_engine_mode = mode_map.get(task_mode, "answer")

                # [FIXED HERE] Passing 'full_input' instead of 'final_prompt'
                # This ensures the AI sees the PDF context
                processed = engine.process(user_id, full_input, level, mode=selected_engine_mode)
                response_text = processed["response"]
                
                st.markdown(response_text)
                
                # Feature 9: Transparency & Metadata
                with st.expander("🔍 Transparency & Metadata"):
                    st.json(processed["metadata"])
                    st.info(f"Task Mode: {task_mode} | Source: {uploaded_pdf.name if uploaded_pdf else 'Direct Input'}")

                # Feature 11: Feedback
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        engine.memory.update_user_level(user_id, 80)
                        st.toast("Feedback recorded: Score increased!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        engine.memory.update_user_level(user_id, 20)
                        st.toast("Feedback recorded: We will adjust the level.")

                # Save Response
                st.session_state.messages.append({"role": "assistant", "content": response_text})