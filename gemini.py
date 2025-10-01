import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
from dotenv import load_dotenv
import io
import fitz  # PyMuPDF for highlighting
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("🚨 GOOGLE_API_KEY not set. Please set it in .env.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- Utility: Clean AI Text (remove HTML gibberish) ---
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = text.replace("&nbsp;", " ")
    return text.strip()

# --- Study Buddy Function ---
def gemini_study_buddy(question, level="standard", age=None):
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""You are a helpful AI tutor. Explain the following question in a clear, step-by-step manner.

Question: {question}

{"Use simple language suitable for a 10-year-old." if level == "simple" else ""}
{"Use simple language suitable for a " + str(age) + "-year-old." if (level == "age_based" and age is not None) else ""}
"""
    response = model.generate_content(prompt)
    return clean_text(response.text)

# --- PDF processing --- 
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception as e:
            # maybe PDF page cannot be parsed
            continue
    return text

def summarize_pdf(pdf_file):
    raw_text = extract_text_from_pdf(pdf_file)
    # handle very large texts by cutting or summarizing in chunks
    max_chunk = 15000
    text_for_model = raw_text[:max_chunk]
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"Summarize the following text into clear study notes:\n\n{text_for_model}"
    response = model.generate_content(prompt)
    summary = clean_text(response.text)
    return summary, raw_text

# --- Ask Question from PDF ---
def ask_pdf_question(pdf_text, question):
    # Highlight relevant PDF sections
    highlighted_snippets = []
    # Simple heuristic: search the snippet (case-insensitive) in text
    lower = pdf_text.lower()
    qlower = question.lower()
    # Find indices of matching parts
    spots = []
    # split into sentences roughly
    sentences = re.split(r'(?<=[.!?]) +', pdf_text)
    for sent in sentences:
        if qlower in sent.lower():
            highlighted_snippets.append(sent.strip())
    # If none found, maybe model provides text anyway
    if not highlighted_snippets:
        highlighted_snippets = ["(No exact matches found for question in text)"]
    # Now generate the answer using model
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
You are a helpful AI tutor. Use ONLY the following PDF text to answer the question.

PDF Content:
{pdf_text[:15000]}  # limit so it doesn't exceed tokens

Question: {question}
"""
    response = model.generate_content(prompt)
    answer = clean_text(response.text)
    return answer, highlighted_snippets

# --- Chat export ---
def export_chat(history, format="txt"):
    """Return bytes of chat history in format txt or md."""
    lines = []
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        if format == "md":
            if role == "user":
                lines.append(f"**You**: {content}")
            else:
                lines.append(f"**AI**: {content}")
        else:
            # txt
            if role == "user":
                lines.append(f"You: {content}")
            else:
                lines.append(f"AI: {content}")
    out = "\n".join(lines)
    return out.encode('utf-8')

# --- UI / Streamlit App ---

st.set_page_config(page_title="AI Study Buddy", page_icon="📘", layout="centered")

st.title("📘 AI Study Buddy")
st.subheader("Learn anything — explained step-by-step by AI")

# Session state
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []  # list of {"role":..., "content":...}
if "explain_age" not in st.session_state:
    st.session_state.explain_age = 10  # default

# Tabs
tab1, tab2 = st.tabs(["❓ Ask a Question", "📄 Summarize & Chat with PDF"])

# --- Tab 1: Ask a Question ---
with tab1:
    question = st.text_area("🔍 What do you want help with?", height=150)
    explain_level = st.selectbox("📚 Explanation level", ["Standard", "Simple (like you're 10)", "By Age"])
    age = None
    if explain_level == "By Age":
        age = st.slider("Select age for explanation", min_value=5, max_value=18, value=10)
    if st.button("Get Help ✨"):
        if not question.strip():
            st.warning("Please write a question before clicking.")
        else:
            with st.spinner("Thinking..."):
                level = "standard"
                if explain_level == "Simple (like you're 10)":
                    level = "simple"
                elif explain_level == "By Age":
                    level = "age_based"
                resp = gemini_study_buddy(question, level=level, age=age)
                st.markdown("### 🧠 Explanation:")
                st.write(resp)

# --- Tab 2: Summarize + Chat with PDF ---
with tab2:
    pdf_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    if pdf_file is not None:
        # show file info
        file_details = {
            "file_name": pdf_file.name,
            "file_size_kb": round(len(pdf_file.getvalue()) / 1024, 2)
        }
        st.write(f"**Uploaded File:** {file_details['file_name']} ({file_details['file_size_kb']} KB)")

    if st.button("Summarize PDF 📄") and pdf_file is not None:
        with st.spinner("Summarizing..."):
            summary, raw_text = summarize_pdf(pdf_file)
            st.session_state.pdf_summary = summary
            st.session_state.pdf_text = raw_text
            st.session_state.pdf_chat_history = []  # reset chat

    # Show summary
    if st.session_state.pdf_summary:
        st.markdown("### 📌 Summary:")
        st.write(st.session_state.pdf_summary)

    # Chat mode for PDF
    if st.session_state.pdf_text:
        st.markdown("---")
        st.subheader("💬 Chat with This PDF")

        # Display chat history
        for entry in st.session_state.pdf_chat_history:
            if entry["role"] == "user":
                st.markdown(f"**🧑 You:** {entry['content']}")
            else:
                st.markdown(f"**🤖 AI:** {entry['content']}")

        pdf_question = st.text_input("Ask a question about the PDF:")
        if st.button("Ask PDF ✨") and pdf_question:
            with st.spinner("Thinking..."):
                answer, snippets = ask_pdf_question(st.session_state.pdf_text, pdf_question)
                st.session_state.pdf_chat_history.append({"role": "user", "content": pdf_question})
                st.session_state.pdf_chat_history.append({"role": "ai", "content": answer})
                # show highlighted snippets
                st.markdown("**🔍 Relevant Snippets from PDF:**")
                for sn in snippets:
                    st.markdown(f"> {sn}")

                st.rerun()

        # Export chat history buttons
        if st.session_state.pdf_chat_history:
            st.markdown("---")
            st.subheader("📂 Export Chat History")
            txt_data = export_chat(st.session_state.pdf_chat_history, format="txt")
            md_data = export_chat(st.session_state.pdf_chat_history, format="md")
            st.download_button("Download as .txt", data=txt_data, file_name="chat_history.txt", mime="text/plain")
            st.download_button("Download as .md", data=md_data, file_name="chat_history.md", mime="text/markdown")

        # Highlighting in PDF view (optional): allow user to see the PDF page with highlighted part
        st.markdown("---")
        st.subheader("🔦 Highlighted PDF Snippet Preview")
        # Simple input to highlight a keyword
        keyword = st.text_input("Enter keyword to highlight in PDF preview:")
        if keyword:
            try:
                # Use PyMuPDF to load PDF and render page(s) with highlighted boxes
                pdf_bytes = pdf_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                # Preview first page or allow navigation
                page_num = st.number_input("Page number", min_value=1, max_value=doc.page_count, value=1, step=1)
                page = doc.load_page(page_num - 1)
                areas = page.search_for(keyword)
                for area in areas:
                    # add rectangle highlight (annotation)
                    highlight = page.add_highlight_annot(area)
                # Render page as image
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, use_column_width=True)
            except Exception as e:
                st.error(f"Could not preview highlighting: {e}")

