import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import tempfile
from docx import Document
import os

# Retrieve the API key from Streamlit's secrets management
api_key = st.secrets["HUGGINGFACE_API_KEY"]

# Initialize the QA model and tokenizer with the API key for authentication
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", token=api_key)
model = AutoModelForQuestionAnswering.from_pretrained("tiiuae/falcon-7b", token=api_key)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def extract_text_from_docx(file_path):
    """ Load Word document text using python-docx """
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip() != '']
        full_text = ' '.join(paragraphs)  # Combine into a single string for QA context
        return full_text
    except Exception as e:
        st.error("Failed to read the document. Please check the file format.")
        return None

class ChatBot:
    def __init__(self):
        self.context = None

    def load_document(self, file_input):
        """ Load document and prepare it for QA """
        if file_input is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_input.getvalue())
                file_path = tmp_file.name

            self.context = extract_text_from_docx(file_path)
            if self.context:
                st.success("Document loaded successfully!")
            os.remove(file_path)
        else:
            st.error("Please upload a Word document.")

    def answer_query(self, question):
        """ Use the QA model to answer a question based on the loaded document """
        if self.context is None:
            st.error("Load a document first.")
            return "No document loaded."

        answer = qa_pipeline({'question': question, 'context': self.context})
        return answer['answer']

# Set up the Streamlit UI
st.title('TINT-Bot for QA')
chat_bot = ChatBot()

file_input = st.file_uploader("Upload Word Document", type=['docx'])
if file_input is not None:
    chat_bot.load_document(file_input)

query = st.text_input("Enter your question:")
if query:
    response = chat_bot.answer_query(query)
    st.text(f"Response: {response}")
