# successfull running

# for downloading a pdf -file from the internet

# import requests
# url = "https://www.impromptubook.com/wp-content/uploads/2023/03/impromptu-rh.pdf"
# response = requests.get(url)
# with open("impromptu-rh.pdf", "wb") as pdf_file:
#     pdf_file.write(response.content)
# print("Download complete.")



from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


# location of the pdf .
doc_reader = PdfReader('impromptu-rh.pdf')


# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text)



# Download embeddings from OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()


docsearch = FAISS.from_texts(texts,embeddings)
docsearch.embedding_function


from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

chain = load_qa_chain(OpenAI(),
                      chain_type="stuff") # we are going to stuff all the docs in at once




# query = "what are the content of this document and write each in points?"
# docs = docsearch.similarity_search(query)
# x =chain.run(input_documents=docs, question=query)
# print("query:-->",query)
# print("ans:-->", x)

import streamlit as st

st.title('DocTalker')
# Create a text input box for the user
query = st.text_input('Type your query here...')

# If the user hits enter
if query:
        docs = docsearch.similarity_search(query)
        x =chain.run(input_documents=docs, question=query)
        st.write(x)


# ********* problem and solutions **********

# FAISS --> works with python 3.9.7  not with 3.10 or 3.11 or 
# change your python version in vs code by selected it in botton right corner

# how to run 
# go to terminal pip install -r doctalk_faiss_requirement.txt
# first activate environment then type streamlit run try3.py to run this file 