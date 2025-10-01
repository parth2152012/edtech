```markdown
# AI Study Buddy

AI Study Buddy is a Streamlit-based application that leverages Google's Generative AI to serve as your personal study assistant. It provides clear, step-by-step explanations for user questions, supports PDF summarization and interactive chatting, and even offers voice input and output capabilities.

## Features

- **Ask a Question:**  
  Interact with our AI tutor to get detailed, step-by-step explanations for your study questions.  
  - Uses different explanation levels: standard, simplified (for kids), or age-specific.

- **PDF Summarization & Chat:**  
  Upload a PDF to have it summarized into clear study notes. Start a chat with the PDF content by asking questions, with highlighted snippets to help you pinpoint relevant sections.

- **Multi-PDF Viewer and Comparison:**  
  Upload multiple PDFs to view side-by-side. Navigate through pages, adjust zoom, and even compare documents by summarizing or extracting key differences.

- **Voice Interactions:**  
  Use voice input to ask questions or interact with the PDF content. AI explanations can also be converted to speech for hands-free studying.

- **Image Question Processing (in some versions):**  
  Upload an image of a question and get a detailed explanation based on the visual content.

## Project Structure

```
.gitignore
[1.py](http://_vscodecontentref_/0)             # Streamlit app with voice and image capabilities.
[2.py](http://_vscodecontentref_/1)             # Alternate Streamlit app implementation.
[gemini.py](http://_vscodecontentref_/2)        # Core definitions for the AI study buddy and PDF processing.
[i.md](http://_vscodecontentref_/3)             # Markdown file with project ideas.
[readme.md](http://_vscodecontentref_/4)        # This file.
[requirements.txt](http://_vscodecontentref_/5) # Python dependencies.
[api.env](http://_vscodecontentref_/6)          # Environment file containing the API key.
```

## Setup Instructions

1. **Install Dependencies:**  
   Ensure you have Python installed, then install all required dependencies by running:
   ```sh
   pip install -r requirements.txt
   ```

2. **API Setup:**  
   Create a `.env` file in the root of the project (or use the provided [api.env](c:\Users\User\Desktop\school\api.env) as a guide) and set your Google API key:
   ```env
   GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
   ```

3. **Run the Application:**  
   Start the Streamlit app by running one of the app files. For example:
   ```sh
   streamlit run 1.py
   ```
   or
   ```sh
   streamlit run 2.py
   ```

## Usage

- **Ask a Question:**  
  Type or speak your question in the appropriate tab and get a detailed explanation from the AI tutor.

- **PDF Upload & Chat:**  
  Upload a PDF, summarize the content, and start a conversation to clarify or dive deeper into the content.

- **Multi-PDF Viewer:**  
  Use the multi-PDF viewer tab to load multiple PDFs side-by-side and use the comparison tools to analyze documents.

## Notes

- This project leverages several Python libraries including [Streamlit](https://docs.streamlit.io/), [google-generativeai](https://developers.generative.ai/), [PyPDF2](https://pythonhosted.org/PyPDF2/), and [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/).
- For detailed functionality and further customization, check out the source files: [1.py](c:\Users\User\Desktop\school\1.py), [2.py](c:\Users\User\Desktop\school\2.py), and [gemini.py](c:\Users\User\Desktop\school\gemini.py).

## License

This project is open source. See the LICENSE file for more information.

## Acknowledgements

Special thanks to the developers of the open source libraries and tools used in this project.
```