import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
import io
import fitz  # PyMuPDF for highlighting
import logging
import time
import hashlib
import docx2txt
import json
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
from PIL import Image
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize session state for theme
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Apply theme based on mode
def apply_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }
        .stTextInput, .stTextArea, .stSelectbox {
            background-color: #2C2C2C;
            color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)
        
# --- Utility: Clean AI Text (remove HTML gibberish) ---
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = text.replace("&nbsp;", " ")
    return text.strip()

# Apply theme at startup
apply_theme()

# Check for API key
if not API_KEY:
    st.error("🚨 GOOGLE_API_KEY not set. Please set it in .env file.")
    st.info("Create a .env file in the same directory with content: GOOGLE_API_KEY=your_api_key_here")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=API_KEY)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# --- Utility Functions ---

def clean_text(text):
    """Remove HTML tags and clean up text formatting."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = text.replace("&nbsp;", " ")
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return text.strip()

def validate_input(text, max_length=5000):
    """Validate user input to prevent security issues."""
    if not text or not text.strip():
        return False, "Input cannot be empty"
    if len(text) > max_length:
        return False, f"Input exceeds maximum length of {max_length} characters"
    # Check for potentially harmful patterns
    if re.search(r'<script|javascript:|onerror=|onclick=', text, re.IGNORECASE):
        return False, "Input contains potentially harmful content"
    return True, ""

@lru_cache(maxsize=32)
def cached_api_call(prompt_hash):
    """Retrieve cached API response."""
    cache_file = f"cache_{prompt_hash}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
    return None

def save_to_cache(prompt_hash, response):
    """Save API response to cache."""
    cache_file = f"cache_{prompt_hash}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(response, f)
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

# --- AI Interaction Functions ---

def gemini_study_buddy(question, level="standard", age=None, retry_count=3):
    """
    Generate explanations for study questions with different complexity levels.
    
    Args:
        question (str): The question to explain
        level (str): Complexity level - "standard", "simple", or "age_based"
        age (int, optional): Age for age-based explanations
        retry_count (int): Number of retries on API failure
        
    Returns:
        str: The explanation text
    """
    # Validate input
    valid, message = validate_input(question)
    if not valid:
        return f"Error: {message}"
    
    # Build an improved, more explicit prompt
    detailed_instruction = "You are a helpful AI tutor. Explain the following question in a clear, step-by-step manner."
    if level == "simple":
        detailed_instruction += "Use simple language suitable for a 10-year-old."
    elif level == "age_based" and age is not None:
        detailed_instruction += f" Use simple language suitable for a {age}-year-old."
    
    prompt = f"""You are a highly knowledgeable and friendly AI tutor.
{detailed_instruction}

Question: {question}
"""
    
    # Check cache
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cached_response = cached_api_call(prompt_hash)
    if cached_response:
        logger.info("Using cached response")
        return cached_response
    
    # Make API call with retry logic
    for attempt in range(retry_count):
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            result = clean_text(response.text)  # Changed to use clean_answer instead of clean_text
            
            # Save to cache
            save_to_cache(prompt_hash, result)
            return result
            
        except Exception as e:
            logger.error(f"API call failed (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(2)  # Wait before retrying
            else:
                return f"Sorry, I couldn't process your request after {retry_count} attempts. Error: {str(e)}"

# --- File Processing Functions ---

def extract_text_from_file(uploaded_file):
    """
    Extract text from various file formats.
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        str: Extracted text content
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            return docx2txt.process(uploaded_file)
        elif file_type == 'txt':
            return uploaded_file.getvalue().decode('utf-8')
        else:
            return f"Unsupported file format: {file_type}"
    except Exception as e:
        logger.error(f"Error extracting text from {file_type} file: {e}")
        return f"Error extracting text: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF files with improved error handling."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {i+1} ---\n{page_text}\n\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1}: {e}")
                text += f"--- Page {i+1} ---\n[Content extraction failed]\n\n"
                
        if not text.strip():
            # Try alternative extraction with PyMuPDF if PyPDF2 fails
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
            text = ""
            for i, page in enumerate(doc):
                text += f"--- Page {i+1} ---\n{page.get_text()}\n\n"
                
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise Exception(f"Failed to process PDF: {str(e)}")

def extract_images_from_pdf(pdf_file):
    """Extract images from PDF file."""
    images = []
    try:
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(BytesIO(image_bytes))
                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "image": image
                })
                
        return images
    except Exception as e:
        logger.error(f"Image extraction failed: {e}")
        return []

def process_large_text(text, chunk_size=10000, overlap=1000):
    """Process large text by breaking it into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Find a good breaking point (end of sentence)
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to find sentence end
            sentence_end = text.rfind('. ', start, end) + 1
            if sentence_end > start:
                end = sentence_end
                
        chunks.append(text[start:end])
        start = end - overlap
        
    return chunks

def summarize_pdf(pdf_file):
    """
    Generate a summary of PDF content with improved handling for large documents.
    
    Args:
        pdf_file: The uploaded PDF file
        
    Returns:
        tuple: (summary text, raw text)
    """
    try:
        raw_text = extract_text_from_pdf(pdf_file)
        
        # Process large documents in chunks
        chunks = process_large_text(raw_text)
        summaries = []
        
        # Process each chunk
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            progress_bar.progress((i + 1) / len(chunks))
            
            model = genai.GenerativeModel('gemini-2.5-pro')
            prompt = f"Summarize the following text into clear study notes:\n\n{chunk}"
            
            try:
                response = model.generate_content(prompt)
                chunk_summary = clean_text(response.text)
                summaries.append(chunk_summary)
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i+1}: {e}")
                summaries.append(f"[Failed to summarize this section: {str(e)}]")
                
        # Combine summaries
        if len(summaries) > 1:
            combined_summary = "\n\n".join(summaries)
            
            # Generate a meta-summary if we have multiple chunks
            try:
                model = genai.GenerativeModel('gemini-2.5-pro')
                prompt = f"Create a concise overall summary of these section summaries:\n\n{combined_summary}"
                response = model.generate_content(prompt)
                final_summary = clean_text(response.text)
            except Exception as e:
                logger.error(f"Failed to create meta-summary: {e}")
                final_summary = combined_summary
        else:
            final_summary = summaries[0]
            
        return final_summary, raw_text
        
    except Exception as e:
        logger.error(f"PDF summarization failed: {e}")
        return f"Failed to summarize PDF: {str(e)}", ""

def ask_pdf_question(pdf_text, question, context_window=3):
    """
    Answer questions about PDF content with improved context detection.
    
    Args:
        pdf_text (str): The PDF content
        question (str): The question to answer
        context_window (int): Number of sentences to include before/after matches
        
    Returns:
        tuple: (answer text, highlighted snippets)
    """
    # Validate input
    valid, message = validate_input(question)
    if not valid:
        return f"Error: {message}", []
    
    # Improved context extraction
    highlighted_snippets = []
    qlower = question.lower()
    
    # Split into sentences more accurately
    sentences = re.split(r'(?<=[.!?])\s+', pdf_text)
    
    # Find relevant sentences with context window
    for i, sent in enumerate(sentences):
        if qlower in sent.lower():
            # Get context window
            start = max(0, i - context_window)
            end = min(len(sentences), i + context_window + 1)
            
            # Create context snippet
            context = " ".join(sentences[start:end])
            highlighted_snippets.append(context.strip())
    
    # If no exact matches, try keyword matching
    if not highlighted_snippets:
        keywords = re.sub(r'[^\w\s]', '', qlower).split()
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in keywords if len(keyword) > 3):
                # Get context window
                start = max(0, i - context_window)
                end = min(len(sentences), i + context_window + 1)
                
                # Create context snippet
                context = " ".join(sentences[start:end])
                highlighted_snippets.append(context.strip())
    
    # If still no matches
    if not highlighted_snippets:
        highlighted_snippets = ["(No relevant matches found for question in text)"]
    
    # Process text in chunks if needed
    chunks = process_large_text(pdf_text)
    answers = []
    
    for i, chunk in enumerate(chunks):
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
            prompt = f"""
You are a helpful AI tutor. Use ONLY the following PDF text to answer the question.
If the answer cannot be found in the text, say "I don't see information about that in the document."

PDF Content (Part {i+1}/{len(chunks)}):
{chunk}

Question: {question}
"""
            response = model.generate_content(prompt)
            answers.append(clean_text(response.text))
        except Exception as e:
            logger.error(f"Failed to process chunk {i+1}: {e}")
            answers.append(f"[Error processing this section: {str(e)}]")
    
    # Combine answers if multiple chunks
    if len(answers) > 1:
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
            prompt = f"""
Combine these partial answers into a single coherent response:

{' '.join(answers)}

Question was: {question}
"""
            response = model.generate_content(prompt)
            final_answer = clean_text(response.text)
        except Exception as e:
            logger.error(f"Failed to combine answers: {e}")
            final_answer = " ".join(answers)
    else:
        final_answer = answers[0]
        
    return final_answer, highlighted_snippets

def generate_study_questions(pdf_text, num_questions=5):
    """Generate study questions from PDF content."""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = f"""
Based on the following text, generate {num_questions} thoughtful study questions that would help a student understand the key concepts.
For each question, also provide a brief answer.

Text:
{pdf_text[:15000]}
"""
        response = model.generate_content(prompt)
        return clean_text(response.text)
    except Exception as e:
        logger.error(f"Failed to generate study questions: {e}")
        return f"Failed to generate study questions: {str(e)}"

# --- Chat and Export Functions ---

def export_chat(history, format="txt"):
    """
    Export chat history in various formats.
    
    Args:
        history (list): List of chat messages
        format (str): Export format - "txt", "md", or "json"
        
    Returns:
        bytes: Encoded chat history
    """
    if format == "json":
        return json.dumps(history, indent=2).encode('utf-8')
        
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Chat Export - {timestamp}\n")
    
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

def save_chat_history():
    """Save chat history to session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Add current chat to history if it exists
    if st.session_state.pdf_chat_history:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({
            "timestamp": timestamp,
            "messages": st.session_state.pdf_chat_history.copy(),
            "document": st.session_state.current_document if "current_document" in st.session_state else "Unknown"
        })

# --- UI / Streamlit App ---

st.set_page_config(page_title="AI Study Buddy", page_icon="📘", layout="wide")

# Sidebar for settings and history
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme toggle
    if st.toggle("Dark Mode", value=st.session_state.dark_mode):
        st.session_state.dark_mode = not st.session_state.dark_mode
        apply_theme()
        st.rerun()
    
    # API status
    st.subheader("API Status")
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        st.success("✅ Gemini API Connected")
    except Exception as e:
        st.error(f"❌ API Connection Failed: {str(e)}")
    
    # Clear data
    st.subheader("Data Management")
    if st.button("Clear All Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

# Main content
st.title("📘 AI Study Buddy")
st.subheader("Learn anything — explained step-by-step by AI")

# Initialize session state
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []
if "explain_age" not in st.session_state:
    st.session_state.explain_age = 10
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "extracted_images" not in st.session_state:
    st.session_state.extracted_images = []
if "current_document" not in st.session_state:
    st.session_state.current_document = None

# Tabs
tab1, tab2, tab3 = st.tabs(["❓ Ask a Question", "📄 Document Analysis", "📚 Study History"])

# --- Tab 1: Ask a Question ---
with tab1:
    question = st.text_area("🔍 What do you want help with?", height=150)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        explain_level = st.selectbox("📚 Explanation level", ["Standard", "Simple (like you're 10)", "By Age"])
        age = None
        if explain_level == "By Age":
            age = st.slider("Select age for explanation", min_value=5, max_value=18, value=st.session_state.explain_age)
            st.session_state.explain_age = age
    
    with col2:
        st.write("Previous Questions:")
        if st.session_state.question_history:
            selected_question = st.selectbox(
                "Load previous question", 
                options=[q["question"] for q in st.session_state.question_history],
                label_visibility="collapsed"
            )
            if st.button("Load"):
                # Find the selected question in history
                for item in st.session_state.question_history:
                    if item["question"] == selected_question:
                        # Set the form values
                        question = item["question"]
                        explain_level = item["level"]
                        if explain_level == "By Age":
                            st.session_state.explain_age = item["age"]
                        st.rerun()
    
    if st.button("Get Help ✨", use_container_width=True):
        if not question.strip():
            st.warning("Please write a question before clicking.")
        else:
            with st.spinner("Thinking..."):
                try:
                    level = "standard"
                    if explain_level == "Simple (like you're 10)":
                        level = "simple"
                    elif explain_level == "By Age":
                        level = "age_based"
                    
                    # Save to question history
                    st.session_state.question_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "level": explain_level,
                        "age": age
                    })
                    
                    # Get response
                    resp = gemini_study_buddy(question, level=level, age=age)
                    
                    st.markdown("### 🧠 Explanation:")
                    st.write(resp)
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        export_data = f"Question: {question}\n\nAnswer: {resp}"
                        st.download_button(
                            "Download as Text", 
                            data=export_data.encode('utf-8'),
                            file_name=f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error in study buddy: {e}")

# --- Tab 2: Document Analysis ---
with tab2:
    st.subheader("📄 Upload & Analyze Documents")
    
    # File upload with multiple formats
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "txt"],
        help="Upload a PDF, Word document, or text file to analyze"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_details = {
            "file_name": uploaded_file.name,
            "file_size_kb": round(len(uploaded_file.getvalue()) / 1024, 2),
            "file_type": uploaded_file.name.split('.')[-1].upper()
        }
        st.session_state.current_document = uploaded_file.name
        
        st.write(f"**Uploaded File:** {file_details['file_name']} ({file_details['file_size_kb']} KB)")
        
        # Document analysis options
        analysis_options = st.columns(3)
        
        with analysis_options[0]:
            if st.button("📝 Summarize Document", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        if file_details['file_type'].lower() == 'pdf':
                            summary, raw_text = summarize_pdf(uploaded_file)
                            
                            # Extract images if PDF
                            st.session_state.extracted_images = extract_images_from_pdf(uploaded_file)
                        else:
                            raw_text = extract_text_from_file(uploaded_file)
                            
                            # Summarize non-PDF documents
                            model = genai.GenerativeModel('gemini-2.5-pro')
                            prompt = f"Summarize the following text into clear study notes:\n\n{raw_text[:15000]}"
                            response = model.generate_content(prompt)
                            summary = clean_text(response.text)
                            
                        st.session_state.pdf_summary = summary
                        st.session_state.pdf_text = raw_text
                        st.session_state.pdf_chat_history = []  # reset chat
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        logger.error(f"Document processing error: {e}")
        
        with analysis_options[1]:
            if st.button("❓ Generate Study Questions", use_container_width=True):
                with st.spinner("Generating questions..."):
                    try:
                        if not st.session_state.pdf_text:
                            # Extract text if not already done
                            text = extract_text_from_file(uploaded_file)
                            st.session_state.pdf_text = text
                        
                        questions = generate_study_questions(st.session_state.pdf_text)
                        st.session_state.study_questions = questions
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
                        logger.error(f"Question generation error: {e}")
        
        with analysis_options[2]:
            if st.button("🔍 Extract Images", use_container_width=True):
                with st.spinner("Extracting images..."):
                    try:
                        if file_details['file_type'].lower() == 'pdf':
                            st.session_state.extracted_images = extract_images_from_pdf(uploaded_file)
                            if not st.session_state.extracted_images:
                                st.info("No images found in the document")
                        else:
                            st.info("Image extraction is only available for PDF files")
                    except Exception as e:
                        st.error(f"Error extracting images: {str(e)}")
                        logger.error(f"Image extraction error: {e}")
    
    # Show summary if available
    if st.session_state.pdf_summary:
        st.markdown("### 📌 Document Summary:")
        st.write(st.session_state.pdf_summary)
        
        # Export summary
        st.download_button(
            "Download Summary", 
            data=st.session_state.pdf_summary.encode('utf-8'),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Show study questions if available
    if "study_questions" in st.session_state and st.session_state.study_questions:
        st.markdown("### 📝 Study Questions:")
        st.write(st.session_state.study_questions)
        
        # Export questions
        st.download_button(
            "Download Study Questions", 
            data=st.session_state.study_questions.encode('utf-8'),
            file_name=f"study_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Show extracted images if available
    if st.session_state.extracted_images:
        st.markdown("### 🖼️ Extracted Images:")
        
        # Display images in a grid
        cols = st.columns(3)
        for i, img_data in enumerate(st.session_state.extracted_images):
            with cols[i % 3]:
                st.image(img_data["image"], caption=f"Page {img_data['page']}, Image {img_data['index']+1}")
    
    # Chat mode for document
    if st.session_state.pdf_text:
        st.markdown("---")
        st.subheader("💬 Chat with This Document")
        
        # Display chat history
        chat_container = st.container(height=400, border=True)
        with chat_container:
            for entry in st.session_state.pdf_chat_history:
                if entry["role"] == "user":
                    st.markdown(f"**🧑 You:** {entry['content']}")
                else:
                    st.markdown(f"**🤖 AI:** {entry['content']}")
        
        # Chat input
        pdf_question = st.text_input("Ask a question about the document:")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button("Ask Document ✨", use_container_width=True) and pdf_question:
                with st.spinner("Thinking..."):
                    try:
                        answer, snippets = ask_pdf_question(st.session_state.pdf_text, pdf_question)
                        st.session_state.pdf_chat_history.append({"role": "user", "content": pdf_question})
                        st.session_state.pdf_chat_history.append({"role": "ai", "content": answer})
                        
                        # Save to history
                        save_chat_history()
                        
                        # Show highlighted snippets
                        st.markdown("**🔍 Relevant Snippets from Document:**")
                        for sn in snippets:
                            st.markdown(f"> {sn}")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Document chat error: {e}")
        
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.pdf_chat_history = []
                st.rerun()
        
        # Export chat history buttons
        if st.session_state.pdf_chat_history:
            st.markdown("---")
            st.subheader("📂 Export Chat History")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                txt_data = export_chat(st.session_state.pdf_chat_history, format="txt")
                st.download_button(
                    "Download as .txt", 
                    data=txt_data, 
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                md_data = export_chat(st.session_state.pdf_chat_history, format="md")
                st.download_button(
                    "Download as .md", 
                    data=md_data, 
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            with col3:
                json_data = export_chat(st.session_state.pdf_chat_history, format="json")
                st.download_button(
                    "Download as .json", 
                    data=json_data, 
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        # Highlighting in PDF view
        if uploaded_file and uploaded_file.name.lower().endswith('.pdf'):
            st.markdown("---")
            st.subheader("🔦 Highlighted PDF Snippet Preview")
            
            # Simple input to highlight a keyword
            keyword = st.text_input("Enter keyword to highlight in PDF preview:")
            if keyword:
                try:
                    # Use PyMuPDF to load PDF and render page(s) with highlighted boxes
                    pdf_bytes = uploaded_file.getvalue()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    # Preview first page or allow navigation
                    page_num = st.number_input("Page number", min_value=1, max_value=doc.page_count, value=1, step=1)
                    page = doc.load_page(page_num - 1)
                    
                    # Search for keyword and highlight
                    areas = page.search_for(keyword)
                    for area in areas:
                        # add rectangle highlight (annotation)
                        highlight = page.add_highlight_annot(area)
                        
                    # Render page as image
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, use_column_width=True)
                    
                    # Show count of matches
                    st.info(f"Found {len(areas)} matches for '{keyword}' on page {page_num}")
                except Exception as e:
                    st.error(f"Could not preview highlighting: {e}")
                    logger.error(f"PDF highlighting error: {e}")

# --- Tab 3: Study History ---
with tab3:
    st.subheader("📚 Your Learning History")
    
    # Question history
    if st.session_state.question_history:
        st.markdown("### ❓ Previous Questions")
        
        # Create a dataframe for better display
        questions_data = []
        for q in st.session_state.question_history:
            questions_data.append({
                "Time": q["timestamp"],
                "Question": q["question"],
                "Level": q["level"]
            })
            
        # Display as table
        st.dataframe(questions_data, use_container_width=True)
        
        # Export question history
        if st.button("Export Question History"):
            export_data = "Question History\n\n"
            for q in st.session_state.question_history:
                export_data += f"Time: {q['timestamp']}\n"
                export_data += f"Question: {q['question']}\n"
                export_data += f"Level: {q['level']}\n"
                if q['level'] == 'By Age' and 'age' in q:
                    export_data += f"Age: {q['age']}\n"
                export_data += "\n---\n\n"
                
            st.download_button(
                "Download Question History", 
                data=export_data.encode('utf-8'),
                file_name=f"question_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.info("No question history yet. Ask some questions to build your history!")
    
    # Document chat history
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.markdown("### 📄 Document Chat History")
        
        for i, chat in enumerate(st.session_state.chat_history):
            with st.expander(f"Chat {i+1}: {chat['document']} - {chat['timestamp']}"):
                for msg in chat["messages"]:
                    if msg["role"] == "user":
                        st.markdown(f"**🧑 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {msg['content']}")
                        
                # Export this specific chat
                chat_data = export_chat(chat["messages"], format="txt")
                st.download_button(
                    "Export This Chat", 
                    data=chat_data,
                    file_name=f"chat_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        st.info("No document chat history yet. Chat with some documents to build your history!")
    
    # Clear history options
    st.markdown("---")
    st.subheader("🧹 Clear History")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Question History"):
            st.session_state.question_history = []
            st.success("Question history cleared!")
            st.rerun()
            
    with col2:
        if st.button("Clear Document Chat History"):
            if "chat_history" in st.session_state:
                st.session_state.chat_history = []
            st.success("Document chat history cleared!")
            st.rerun()


#end of the code
