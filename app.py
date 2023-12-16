import os
import streamlit as st
import pickle
import time
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

filepath = 'vectorstore_openai.pkl'

llm = OpenAI(temperature=0.9, max_tokens=500)

st.title("News Research Tool 📋📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    # load data
    main_placeholder.text("Data Loading Started...✅✅✅")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # split data
    main_placeholder.text("Text Splitter Started...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(filepath, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            vectorestore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorestore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer") 
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
            