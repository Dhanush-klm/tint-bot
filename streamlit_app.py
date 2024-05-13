import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from docx import Document
import tempfile
import os

# Retrieve the API key from Streamlit's secrets management
api_key = st.secrets["HUGGINGFACE_API_KEY"]

# Initialize the tokenizer and model for embeddings with the API key for authentication
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", token=api_key)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", token=api_key)

def extract_text_from_docx(file_path):
    """ Load Word document text using python-docx """
    try:
        doc = Document(file_path)
        return [p.text for p in doc.paragraphs if p.text.strip() != '']
    except Exception as e:
        st.error("Failed to read the document. Please check the file format.")
        return None

def embed_text(text_list):
    """ Generate embeddings for a list of text documents """
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output  # Use the pooler output for sentence embeddings
    return embeddings

class ChatBot:
    def __init__(self):
        self.embeddings = None
        self.documents = []

    def load_documents(self, file_input):
        """ Load and embed documents from the uploaded file """
        if file_input is not None:
            # Temporary save file to disk to read with Document
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_input.getvalue())
                file_path = tmp_file.name

            documents = extract_text_from_docx(file_path)
            if documents:
                self.embeddings = embed_text(documents)
                self.documents = documents
                st.success("File loaded and embeddings created!")
            # Clean up the temporary file
            os.remove(file_path)
        else:
            st.error("Please upload a Word document.")

    def answer_query(self, query):
        """ Find the document section most relevant to the query """
        if self.embeddings is None:
            st.error("Load a document file first.")
            return "No document loaded."

        query_emb = embed_text([query])
        cos_sim = torch.nn.functional.cosine_similarity(query_emb, self.embeddings, dim=1)
        top_result_idx = cos_sim.argmax()
        return self.documents[top_result_idx]

# Set up the Streamlit UI
st.title('TINT-Bot')
chat_bot = ChatBot()

file_input = st.file_uploader("Upload Word Document", type=['docx'])
if file_input is not None:
    chat_bot.load_documents(file_input)

query = st.text_input("Enter your question:")
if query:
    response = chat_bot.answer_query(query)
    st.text(f"Response: {response}")
