from providers.OpenRouterProvider import OpenRouterProvider
import pdfplumber


llm = OpenRouterProvider()

def extract_pdf(path):
    """
    Extract clean text from PDF using pdfplumber.
    """
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


def summarize_text(text):
    """
    Summarize long text using chunking + final summary.
    """
    if len(text) < 1500:
        return llm.summarize(text)

    # Otherwise → break into chunks
    chunks = []
    size = 1400

    for i in range(0, len(text), size):
        chunk = text[i:i+size]
        partial_summary = llm.summarize(chunk)
        chunks.append(partial_summary)

    combined = "\n".join(chunks)

    final_summary = llm.summarize(combined)
    return final_summary
