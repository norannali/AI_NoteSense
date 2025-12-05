# 🎓 AI NoteSense: The Intelligent & Ethical Study Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)
![OpenRouter](https://img.shields.io/badge/AI-OpenRouter-7F27FF)
![Status](https://img.shields.io/badge/Status-Prototype-green)

**AI NoteSense** is a Human-AI interaction system designed to transform complex lecture notes into personalized, accessible knowledge. [cite_start]Unlike standard summarizers, it features an **Emotion-Aware Engine** that detects student frustration and adapts its tone to be supportive, acting as an empathetic study partner[cite: 71, 139].

---

## 🚀 Key Features

* [cite_start]**📄 Intelligent PDF Processing**: Extracts text from lecture slides and automatically detects the main topic[cite: 86, 178].
* **🧠 Cognitive Load Reduction**:
    * [cite_start]**Summarizer Agent**: Compresses long texts into concise bullet points[cite: 87, 296].
    * [cite_start]**Visualizer Agent**: Generates dynamic concept maps (Graphviz) to visualize relationships between ideas[cite: 89, 178].
* **🎭 Emotion-Aware & Adaptive**:
    * [cite_start]Detects frustration (e.g., "I'm lost") and switches to a supportive, encouraging tone[cite: 91, 154].
    * [cite_start]**Opt-out**: Users can disable emotional features for a strictly utilitarian experience[cite: 188].
* **💡 "Explain Differently" Engine**:
    * [cite_start]Provides 3 distinct explanation styles: *Simple*, *Example-based*, and *Technical*[cite: 90, 178].
* **🔍 Transparency & Ethics**:
    * [cite_start]**Source Citing**: Every output is linked to its source in the document[cite: 92, 134].
    * [cite_start]**Data Privacy**: Session-based architecture ensures uploaded files are deleted immediately after use [cite: 119-121].

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/) (Interactive Web UI)
* **Backend Logic**: Python
* [cite_start]**AI Models**: OpenRouter API (Accessing models like Mistral 8x7B / Llama 3) [cite: 95]
* **NLP Tools**: `pdfplumber` (Extraction), `re` (Regex pattern matching)
* **Visualization**: `Graphviz` (DOT language rendering)

---

## ⚙️ Installation & Setup

### Prerequisites
1.  **Python 3.10+** installed.
2.  **Graphviz** (System Executable) installed. *This is required for the visual maps.*
    * *Windows*: Download installer from [Graphviz.org](https://graphviz.org/download/) and select **"Add to System PATH"**.
    * *Mac*: `brew install graphviz`
    * *Linux*: `sudo apt-get install graphviz`

### Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/norannali/AI_NoteSense.git](https://github.com/norannali/AI_NoteSense.git)
    cd AI_NoteSense
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**
    * Create a `.env` file in the root directory (or set it in your environment variables).
    * Add your OpenRouter key:
    ```bash
    OPENROUTER_API_KEY=sk-or-v1-your-key-here
    ```

4.  **Run the Application**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📖 How to Use

1.  **Upload**: Upload your lecture PDF in the sidebar. [cite_start]The system will auto-detect the topic[cite: 178].
2.  **Select Task**: Choose a mode from the sidebar:
    * `Summarize Document`: Get a quick overview.
    * `Visual Outline`: Generate a concept map.
    * `Explain Differently`: Get multi-perspective explanations.
    * `General Chat`: Ask specific questions.
3.  **Interact**:
    * Type questions like *"Explain this simply"* or *"I don't understand"*.
    * [cite_start]If you express frustration, notice how the AI's tone changes to be more supportive[cite: 147].
4.  **Feedback**: Use the 👍 / 👎 buttons to rate responses. [cite_start]This adjusts your "Student Level" in the background[cite: 178].

---

## 🎨 Design Principles (Shneiderman’s 8 Rules)

This project strictly adheres to HCI principles:
* [cite_start]**Consistency**: Unified color palette (Academic Blue) and layout[cite: 102].
* [cite_start]**User Control**: Clear "Clear History" and "Undo" actions[cite: 103, 136].
* [cite_start]**Feedback**: Loaders and status messages for every AI action[cite: 103].

---

## 🔮 Future Roadmap

* [cite_start][ ] **Multimodal Input**: Support for handwriting and image recognition in notes[cite: 193].
* [cite_start][ ] **Long-term Memory**: User profiles that track learning progress across semesters[cite: 195].
* [cite_start][ ] **Automated Fact-Checking**: rigorous verification against source texts[cite: 215].

---

    * [Noran Ali](https://github.com/norannali)

---

> **Note**: This project is a prototype designed for educational purposes. API keys and personal data are handled securely via session-state management.
