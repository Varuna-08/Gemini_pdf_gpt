import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize GoogleGenerativeAIEmbeddings outside of functions to avoid re-initialization
embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Remove allow_dangerous_deserialization argument
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, pdf_docs):
    try:
        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text)
        vector_store = get_vector_store(text_chunks)
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error("Error during question answering.")
        st.error(str(e))

def main():
    st.set_page_config(page_title="Chat With Multiple PDF", page_icon="📚")
    st.header("💁 Chat with PDF using Gemini")

    pdf_docs = None  # Initialize pdf_docs variable

    user_question = st.text_input("Ask the questions from the PDF Files")

    with st.sidebar:
        st.title("Upload PDF Files 📂")
        pdf_docs = st.file_uploader("Upload your 📄 PDF files and click on 'Submit & Process' to generate the FAISS index.", accept_multiple_files=True)
        if st.button("Submit & Process 🚀"):
            if pdf_docs:
                with st.spinner("Processing.. ⌛️"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    try:
                        get_vector_store(text_chunks)
                        st.success("✅ Done")
                    except Exception as e:
                        st.error(f"Error generating FAISS index: {e}")
            else:
                st.error("Please upload PDF files")

    if user_question and pdf_docs:
        user_input(user_question, pdf_docs)
    elif user_question and pdf_docs is None:
        st.warning("Please upload PDF files and click 'Submit & Process' to generate the FAISS index.")

if __name__ == "__main__":
    main()
