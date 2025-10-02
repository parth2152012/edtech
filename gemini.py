import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
from dotenv import load_dotenv
import io
import fitz  # PyMuPDF for highlighting
from datetime import datetime
import PIL.Image
from deep_translator import GoogleTranslator
import base64  # For audio playback
import gtts  # Google Text-to-Speech
import speech_recognition as sr  # For speech-to-text
import tempfile
import numpy as np
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder



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

# --- Text-to-Speech Function ---
def text_to_speech(text, lang='en'):
    try:
        tts = gtts.gTTS(text=text, lang=lang)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# --- Audio Player HTML ---
def get_audio_player_html(audio_data):
    if audio_data is None:
        return ""
    b64 = base64.b64encode(audio_data).decode()
    return f"""
    <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """

# --- Speech-to-Text Function ---
def speech_to_text(audio_bytes, language="en-US"):
    try:
        # Save audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio_path = temp_audio.name
            # If the audio is in webm format, convert it to wav
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio.export(temp_audio_path, format="wav")
        
        # Use speech recognition to convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
        
        # Clean up the temporary file
        os.unlink(temp_audio_path)
        
        return text
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return None

# --- Language Translation Functions ---
def translate_text(text, target_language):
    return GoogleTranslator(source='auto', target=target_language).translate(text)

def translate_to_english(text, source_language):
    if source_language == "en":
        return text
    return GoogleTranslator(source=source_language, target="en").translate(text)

# --- Study Buddy Function ---
def gemini_study_buddy(question, level="standard", age=None, language="en"):
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # Translate question to English if needed
    question_en = question
    if language != "en":
        question_en = translate_to_english(question, language)
    
    prompt = f"""You are a helpful AI tutor. Explain the following question in a clear, step-by-step manner.

Question: {question_en}

{"Use simple language suitable for a 10-year-old." if level == "simple" else ""}
{"Use simple language suitable for a " + str(age) + "-year-old." if (level == "age_based" and age is not None) else ""}
"""
    response = model.generate_content(prompt)
    
    # If there are multiple parts, join them, otherwise use response.text directly.
    if hasattr(response, "parts") and len(response.parts) > 1:
        answer = clean_text("".join(part.text for part in response.parts if hasattr(part, "text")))
    else:
        answer = clean_text(response.text)
    
    # Translate answer back to target language if needed
    if language != "en":
        answer = translate_text(answer, language)
    
    return answer

# --- Image Question Function ---
def process_image_question(image, question, level="standard", age=None, language="en"):
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # Translate question to English if needed
    question_en = question
    if language != "en":
        question_en = translate_to_english(question, language)
    
    prompt = f"""You are a helpful AI tutor. Look at the image and answer the following question in a clear, step-by-step manner.

Question: {question_en}

{"Use simple language suitable for a 10-year-old." if level == "simple" else ""}
{"Use simple language suitable for a " + str(age) + "-year-old." if (level == "age_based" and age is not None) else ""}
"""
    response = model.generate_content([prompt, image])
    answer = clean_text(response.text)
    
    # Translate answer back to target language if needed
    if language != "en":
        answer = translate_text(answer, language)
    
    return answer

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

def summarize_pdf(pdf_file, language="en"):
    raw_text = extract_text_from_pdf(pdf_file)
    # handle very large texts by cutting or summarizing in chunks
    max_chunk = 15000
    text_for_model = raw_text[:max_chunk]
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"Summarize the following text into clear study notes:\n\n{text_for_model}"
    response = model.generate_content(prompt)
    summary = clean_text(response.text)
    
    # Translate summary if needed
    if language != "en":
        summary = translate_text(summary, language)
    
    return summary, raw_text

# --- Ask Question from PDF ---
def ask_pdf_question(pdf_text, question, language="en"):
    # Translate question to English if needed
    question_en = question
    if language != "en":
        question_en = translate_to_english(question, language)
    
    # Highlight relevant PDF sections
    highlighted_snippets = []
    # Simple heuristic: search the snippet (case-insensitive) in text
    lower = pdf_text.lower()
    qlower = question_en.lower()
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

Question: {question_en}
"""
    response = model.generate_content(prompt)
    answer = clean_text(response.text)
    
    # Translate answer back to target language if needed
    if language != "en":
        answer = translate_text(answer, language)
    
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
if "question_history" not in st.session_state:
    st.session_state.question_history = []  # list of {"question":..., "answer":..., "timestamp":...}
if "current_audio" not in st.session_state:
    st.session_state.current_audio = None
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""

# Language selection
languages = {
    "en": "English",
    "mr": "Marathi (मराठी)",
    "hi": "Hindi (हिंदी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)",
    "zh-cn": "Chinese (中文)"
}

# Language code mapping for TTS (some languages might have different codes for TTS)
tts_language_codes = {
    "en": "en",
    "mr": "mr",
    "hi": "hi",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
    "zh-cn": "zh-CN"
}

# Language code mapping for STT (Speech-to-Text)
stt_language_codes = {
    "en": "en-US",
    "mr": "mr-IN",
    "hi": "hi-IN",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "ja": "ja-JP",
    "zh-cn": "zh-CN"
}

selected_language = st.sidebar.selectbox(
    "🌐 Select Language",
    options=list(languages.keys()),
    format_func=lambda x: languages[x]
)

# Tabs
tab1, tab2, tab3 = st.tabs(["❓ Ask a Question", "📷 Ask with Image", "📄 Summarize & Chat with PDF"])

# --- Tab 1: Ask a Question ---
with tab1:
    # Voice input section
    st.markdown("### 🎤 Voice Input")
    
    # Audio recorder for voice input
    st.markdown("Click to record your question")
    audio_bytes = audio_recorder(key="question_tab_recorder")
    
    if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
        st.session_state.last_audio_bytes = audio_bytes
        with st.spinner("Transcribing your voice..."):
            # Convert speech to text
            stt_lang = stt_language_codes.get(selected_language, "en-US")
            transcribed_text = speech_to_text(audio_bytes, language=stt_lang)
            
            if transcribed_text:
                st.session_state.voice_input = transcribed_text
                st.success(f"Transcribed: {transcribed_text}")
            else:
                st.error("Could not transcribe audio. Please try again.")
    
    # Text input with voice input pre-filled
    question = st.text_area("🔍 What do you want help with?", 
                           value=st.session_state.voice_input, 
                           height=150)
    
    # Clear voice input after it's been used
    if question != st.session_state.voice_input and st.session_state.voice_input:
        st.session_state.voice_input = ""
    
    explain_level = st.selectbox("📚 Explanation level", ["Standard", "Simple (like you're 10)", "By Age"])
    age = None
    if explain_level == "By Age":
        age = st.slider("Select age for explanation", min_value=5, max_value=18, value=10)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Get Help ✨"):
            if not question.strip():
                st.warning("Please write or speak a question before clicking.")
            else:
                with st.spinner("Thinking..."):
                    level = "standard"
                    if explain_level == "Simple (like you're 10)":
                        level = "simple"
                    elif explain_level == "By Age":
                        level = "age_based"
                    resp = gemini_study_buddy(question, level=level, age=age, language=selected_language)
                    st.markdown("### 🧠 Explanation:")
                    st.write(resp)
                    
                    # Generate audio for the response
                    tts_lang = tts_language_codes.get(selected_language, "en")
                    audio_data = text_to_speech(resp, lang=tts_lang)
                    st.session_state.current_audio = audio_data
                    
                    # Add hear button
                    if audio_data:
                        st.markdown("### 🔊 Listen to explanation:")
                        st.markdown(get_audio_player_html(audio_data), unsafe_allow_html=True)
                    
                    # Save to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.question_history.append({
                        "question": question,
                        "answer": resp,
                        "timestamp": timestamp,
                        "type": "text",
                        "audio": audio_data
                    })
    
    # History section
    st.markdown("---")
    st.subheader("📜 Question History")
    
    if not st.session_state.question_history:
        st.info("Your question history will appear here.")
    else:
        # Show history in reverse chronological order (newest first)
        for i, item in enumerate(reversed(st.session_state.question_history)):
            with st.expander(f"Q: {item['question'][:50]}... ({item['timestamp']})"):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown(f"**Time:** {item['timestamp']}")
                st.markdown(f"**Type:** {item['type']}")
                
                # Add hear button for history items
                if "audio" in item and item["audio"]:
                    st.markdown("**🔊 Listen:**")
                    st.markdown(get_audio_player_html(item["audio"]), unsafe_allow_html=True)
                elif item["answer"]:  # Generate audio on-demand if not stored
                    if st.button(f"🔊 Generate Audio", key=f"gen_audio_{i}"):
                        tts_lang = tts_language_codes.get(selected_language, "en")
                        audio_data = text_to_speech(item["answer"], lang=tts_lang)
                        if audio_data:
                            st.markdown(get_audio_player_html(audio_data), unsafe_allow_html=True)
                            # Update history with audio
                            item["audio"] = audio_data
        
        # Export history buttons
        st.markdown("---")
        history_txt = "\n\n".join([
            f"Time: {item['timestamp']}\nQuestion: {item['question']}\nAnswer: {item['answer']}\nType: {item['type']}\n"
            for item in st.session_state.question_history
        ])
        st.download_button(
            "Download History as .txt",
            data=history_txt.encode('utf-8'),
            file_name=f"question_history_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.question_history = []
            st.rerun()

# --- Tab 2: Ask with Image ---
with tab2:
    st.markdown("### 📷 Upload an Image of Your Question")
    
    # Voice input for image tab
    st.markdown("### 🎤 Voice Input for Image Question")
    
    # Audio recorder for voice input in image tab
    img_audio_bytes = audio_recorder(key="image_tab_recorder")
    
    if img_audio_bytes and img_audio_bytes != st.session_state.get("last_img_audio_bytes"):
        st.session_state.last_img_audio_bytes = img_audio_bytes
        with st.spinner("Transcribing your voice..."):
            # Convert speech to text
            stt_lang = stt_language_codes.get(selected_language, "en-US")
            img_transcribed_text = speech_to_text(img_audio_bytes, language=stt_lang)
            
            if img_transcribed_text:
                st.session_state.img_voice_input = img_transcribed_text
                st.success(f"Transcribed: {img_transcribed_text}")
            else:
                st.error("Could not transcribe audio. Please try again.")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Text input with voice input pre-filled for image
    if "img_voice_input" not in st.session_state:
        st.session_state.img_voice_input = ""
        
    image_question = st.text_area(
        "Your question about the image:", 
        value=st.session_state.img_voice_input, 
        height=100
    )
    
    # Clear voice input after it's been used
    if image_question != st.session_state.img_voice_input and st.session_state.img_voice_input:
        st.session_state.img_voice_input = ""
    
    explain_level_img = st.selectbox("📚 Explanation level for image", ["Standard", "Simple (like you're 10)", "By Age"], key="img_level")
    age_img = None
    if explain_level_img == "By Age":
        age_img = st.slider("Select age for explanation", min_value=5, max_value=18, value=10, key="img_age")
    
    if st.button("Get Help with Image ✨"):
        if uploaded_image is None:
            st.warning("Please upload an image first.")
        elif not image_question.strip():
            st.warning("Please write or speak a question about the image.")
        else:
            with st.spinner("Analyzing image and thinking..."):
                # Process the image
                image = PIL.Image.open(uploaded_image)
                
                level = "standard"
                if explain_level_img == "Simple (like you're 10)":
                    level = "simple"
                elif explain_level_img == "By Age":
                    level = "age_based"
                
                resp = process_image_question(image, image_question, level=level, age=age_img, language=selected_language)
                
                # Generate audio for the response
                tts_lang = tts_language_codes.get(selected_language, "en")
                audio_data = text_to_speech(resp, lang=tts_lang)
                
                # Display the image and response
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    st.markdown("### 🧠 Explanation:")
                    st.write(resp)
                    
                    # Add hear button
                    if audio_data:
                        st.markdown("### 🔊 Listen to explanation:")
                        st.markdown(get_audio_player_html(audio_data), unsafe_allow_html=True)
                
                # Save to history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Save image to bytes for history
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                
                st.session_state.question_history.append({
                    "question": image_question,
                    "answer": resp,
                    "timestamp": timestamp,
                    "type": "image",
                    "audio": audio_data
                })

# --- Tab 3: Summarize + Chat with PDF ---
with tab3:
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
            summary, raw_text = summarize_pdf(pdf_file, language=selected_language)
            st.session_state.pdf_summary = summary
            st.session_state.pdf_text = raw_text
            st.session_state.pdf_chat_history = []  # reset chat

    # Show summary
    if st.session_state.pdf_summary:
        st.markdown("### 📌 Summary:")
        st.write(st.session_state.pdf_summary)
        
        # Generate audio for the summary
        if st.button("🔊 Listen to Summary"):
            tts_lang = tts_language_codes.get(selected_language, "en")
            audio_data = text_to_speech(st.session_state.pdf_summary, lang=tts_lang)
            if audio_data:
                st.markdown(get_audio_player_html(audio_data), unsafe_allow_html=True)

    # Chat mode for PDF
    if st.session_state.pdf_text:
        st.markdown("---")
        st.subheader("💬 Chat with This PDF")

        # Display chat history
        for i, entry in enumerate(st.session_state.pdf_chat_history):
            if entry["role"] == "user":
                st.markdown(f"**🧑 You:** {entry['content']}")
            else:
                st.markdown(f"**🤖 AI:** {entry['content']}")
                # Add hear button for AI responses
                if "audio" in entry and entry["audio"]:
                    st.markdown(get_audio_player_html(entry["audio"]), unsafe_allow_html=True)
                else:
                    if st.button(f"🔊 Listen", key=f"pdf_audio_{i}"):
                        tts_lang = tts_language_codes.get(selected_language, "en")
                        audio_data = text_to_speech(entry["content"], lang=tts_lang)
                        if audio_data:
                            st.markdown(get_audio_player_html(audio_data), unsafe_allow_html=True)
                            # Update history with audio
                            entry["audio"] = audio_data

        # Voice input for PDF questions
        st.markdown("### 🎤 Voice Input for PDF Question")
        
        # Audio recorder for voice input in PDF tab
        pdf_audio_bytes = audio_recorder(key="pdf_tab_recorder")
        
        if pdf_audio_bytes and pdf_audio_bytes != st.session_state.get("last_pdf_audio_bytes"):
            st.session_state.last_pdf_audio_bytes = pdf_audio_bytes
            with st.spinner("Transcribing your voice..."):
                # Convert speech to text
                stt_lang = stt_language_codes.get(selected_language, "en-US")
                pdf_transcribed_text = speech_to_text(pdf_audio_bytes, language=stt_lang)
                
                if pdf_transcribed_text:
                    st.session_state.pdf_voice_input = pdf_transcribed_text
                    st.success(f"Transcribed: {pdf_transcribed_text}")
                else:
                    st.error("Could not transcribe audio. Please try again.")
        
        # Text input with voice input pre-filled for PDF
        if "pdf_voice_input" not in st.session_state:
            st.session_state.pdf_voice_input = ""
            
        pdf_question = st.text_input(
            "Ask a question about the PDF:", 
            value=st.session_state.pdf_voice_input
        )
        
        # Clear voice input after it's been used
        if pdf_question != st.session_state.pdf_voice_input and st.session_state.pdf_voice_input:
            st.session_state.pdf_voice_input = ""

        if st.button("Ask PDF ✨") and pdf_question:
            with st.spinner("Thinking..."):
                answer, snippets = ask_pdf_question(st.session_state.pdf_text, pdf_question, language=selected_language)
                
                # Generate audio for the response
                tts_lang = tts_language_codes.get(selected_language, "en")
                audio_data = text_to_speech(answer, lang=tts_lang)
                
                st.session_state.pdf_chat_history.append({"role": "user", "content": pdf_question})
                st.session_state.pdf_chat_history.append({"role": "ai", "content": answer, "audio": audio_data})
                
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
