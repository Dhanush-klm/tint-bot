import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from docx import Document
import tempfile
import datetime

# Set up tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def load_db(file_input):
    if file_input is not None:
        # Use a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(file_input.read())
            file_path = tmp_file.name
    else:
        raise ValueError("No file uploaded")

    # Load Word document text
    doc = Document(file_path)
    documents = [p.text for p in doc.paragraphs if p.text.strip() != '']
    
    # Encode documents
    encoded_input = tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)  # Average pooling
    
    return embeddings, documents

class ChatBot:
    def __init__(self):
        self.embeddings = None
        self.documents = []
        self.history = []

    def load_db(self, file_input):
        self.embeddings, self.documents = load_db(file_input)
        st.success("File loaded and embeddings created!")

    def search_documents(self, query):
        if self.embeddings is None:
            st.error("Load a document file first.")
            return

        query_enc = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            query_emb = model(**query_enc).last_hidden_state.mean(dim=1)

        # Compute similarities (cosine)
        cos_sim = torch.nn.functional.cosine_similarity(query_emb, self.embeddings, dim=1)
        top_result_idx = cos_sim.argmax()
        return self.documents[top_result_idx]

    def answer_query(self, query):
        response = self.search_documents(query)
        self.history.append((query, response))
        return response

# Streamlit UI setup
st.title('Document-Based Q&A with Hugging Face')
chat_bot = ChatBot()

file_input = st.file_uploader("Upload Word Document", type=['docx'])
if file_input is not None:
    chat_bot.load_db(file_input)

query = st.text_input("Ask a question:")
if query:
    response = chat_bot.answer_query(query)
    st.text(f"Response: {response}")
