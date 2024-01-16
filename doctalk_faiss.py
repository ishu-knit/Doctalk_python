
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS    
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
import os   

def main():
    st.title("Document Search and Question Answering")

    # File upload widget
    doc_reader = st.file_uploader("Upload a PDF file", type=["pdf"])

    if doc_reader:
        raw_text = ''
        for i, page in enumerate(PdfReader(doc_reader).pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # Splitting up the text into smaller chunks for indexing
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        os.environ["OPENAI_API_KEY"] = "sk-vhCRSEjDhpwpTd2U4CfT3BlbkFJ1OCSQxoS45pa3IENBXCy"
        embeddings = OpenAIEmbeddings()

        docsearch = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        query = st.text_input('Type your query here... then press enter')

        if query:
            docs = docsearch.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            st.write(result)

main()

# # ********* problem and solutions **********

# # FAISS --> works with python 3.9.7  not with 3.10 or 3.11 or 
# # change your python version in vs code by selected it in botton right corner

# # how to run
# # create env by python -m venv <name>
# # go to terminal pip install -r doctalk_faiss_requirement.txt
# # download pdf file 
# # first activate environment then type streamlit run try3.py to run this file 
