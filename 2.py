import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
from dotenv import load_dotenv
import io
import fitz  # PyMuPDF for highlighting
from datetime import datetime
import base64

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

# --- Multi-PDF Viewer Functions ---
def display_pdf(pdf_file, page_num=1, zoom=1.5):
    """Display a specific page of a PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        page = doc.load_page(page_num - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        return img_data
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        return None

def get_pdf_page_count(pdf_file):
    """Get the total number of pages in a PDF."""
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        return doc.page_count
    except Exception as e:
        st.error(f"Error getting page count: {e}")
        return 0

# --- UI / Streamlit App ---

st.set_page_config(page_title="AI Study Buddy", page_icon="📘", layout="wide")

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
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []
if "pdf_viewer_page" not in st.session_state:
    st.session_state.pdf_viewer_page = {}
if "pdf_viewer_zoom" not in st.session_state:
    st.session_state.pdf_viewer_zoom = {}

# Tabs
tab1, tab2, tab3 = st.tabs(["❓ Ask a Question", "📄 Summarize & Chat with PDF", "📚 Multi-PDF Viewer"])

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
    pdf_file = st.file_uploader("📂 Upload a PDF", type=["pdf"], key="pdf_uploader")
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

# --- Tab 3: Multi-PDF Viewer ---
with tab3:
    st.header("📚 Multi-PDF Viewer")
    st.write("Upload and view multiple PDFs side by side")
    
    # PDF uploader for multiple files
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="multi_pdf_uploader")
    
    if uploaded_files:
        # Store uploaded files in session state
        st.session_state.uploaded_pdfs = uploaded_files
        
        # Display controls for each PDF
        num_pdfs = len(uploaded_files)
        cols = st.columns(min(num_pdfs, 3))  # Max 3 columns
        
        for i, pdf_file in enumerate(uploaded_files):
            col_idx = i % 3
            with cols[col_idx]:
                st.subheader(f"📄 {pdf_file.name}")
                
                # Initialize page and zoom for this PDF if not exists
                if pdf_file.name not in st.session_state.pdf_viewer_page:
                    st.session_state.pdf_viewer_page[pdf_file.name] = 1
                if pdf_file.name not in st.session_state.pdf_viewer_zoom:
                    st.session_state.pdf_viewer_zoom[pdf_file.name] = 1.5
                
                # Get page count
                page_count = get_pdf_page_count(pdf_file)
                
                # Page navigation
                if page_count > 1:
                    page_num = st.number_input(
                        f"Page", 
                        min_value=1, 
                        max_value=page_count, 
                        value=st.session_state.pdf_viewer_page[pdf_file.name],
                        key=f"page_{pdf_file.name}"
                    )
                    st.session_state.pdf_viewer_page[pdf_file.name] = page_num
                
                # Zoom control
                zoom_level = st.slider(
                    "Zoom", 
                    min_value=0.5, 
                    max_value=3.0, 
                    value=st.session_state.pdf_viewer_zoom[pdf_file.name],
                    step=0.1,
                    key=f"zoom_{pdf_file.name}"
                )
                st.session_state.pdf_viewer_zoom[pdf_file.name] = zoom_level
                
                # Display the PDF page
                img_data = display_pdf(
                    pdf_file, 
                    page_num=st.session_state.pdf_viewer_page[pdf_file.name], 
                    zoom=st.session_state.pdf_viewer_zoom[pdf_file.name]
                )
                
                if img_data:
                    st.image(img_data, use_column_width=True)
                    st.caption(f"Page {st.session_state.pdf_viewer_page[pdf_file.name]} of {page_count}")
                
                # Quick actions for each PDF
                st.download_button(
                    "Download Text", 
                    data=extract_text_from_pdf(pdf_file).encode('utf-8'),
                    file_name=f"{pdf_file.name}_text.txt",
                    mime="text/plain",
                    key=f"download_{pdf_file.name}"
                )
    else:
        st.info("Upload PDF files to use the multi-PDF viewer")
    
    # Compare functionality
    if len(st.session_state.uploaded_pdfs) > 1:
        st.markdown("---")
        st.subheader("🔍 Compare PDFs")
        
        compare_option = st.selectbox(
            "What would you like to compare?",
            ["Select an option", "Summarize all", "Find common themes", "Extract key differences"]
        )
        
        if compare_option != "Select an option":
            with st.spinner("Analyzing documents..."):
                try:
                    # Extract text from all PDFs
                    pdf_texts = []
                    for pdf_file in st.session_state.uploaded_pdfs:
                        pdf_text = extract_text_from_pdf(pdf_file)
                        pdf_texts.append(f"--- {pdf_file.name} ---\n{pdf_text[:5000]}\n\n")
                    
                    combined_text = "\n".join(pdf_texts)
                    
                    # Create appropriate prompt based on selection
                    if compare_option == "Summarize all":
                        prompt = f"Provide a concise summary of each of these documents:\n\n{combined_text}"
                    elif compare_option == "Find common themes":
                        prompt = f"Identify common themes across these documents:\n\n{combined_text}"
                    elif compare_option == "Extract key differences":
                        prompt = f"Extract the key differences between these documents:\n\n{combined_text}"
                    
                    # Get comparison from AI
                    model = genai.GenerativeModel('gemini-2.5-pro')
                    response = model.generate_content(prompt)
                    comparison = clean_text(response.text)
                    
                    st.markdown("### 📊 Comparison Results")
                    st.write(comparison)
                    
                except Exception as e:
                    st.error(f"Error comparing documents: {e}")