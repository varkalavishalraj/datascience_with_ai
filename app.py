import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Ensure Document is imported

# Load environment variables from .env file
load_dotenv()

# Retrieve Cohere API key
COHERE_API_KEY = os.getenv("wHJ7F1CsKJjZWmA4Et2xNpSdjDXkriJdgf3LICkl")
if not COHERE_API_KEY:
    raise ValueError("Cohere API key is missing. Ensure it is set in the .env file or environment variables.")

# Initialize the Cohere model
llm = Cohere(model="command-xlarge-nightly", cohere_api_key=COHERE_API_KEY)

# Streamlit app layout
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Sidebar inputs for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs", key="process_urls_button")
main_placeholder = st.empty()

# Path to store FAISS index
faiss_store_path = "faiss_store"

# Processing step
if process_url_clicked:
    try:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading... Started ✅")
        data = loader.load()

        # Split data into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting... Started ✅")
        docs = text_splitter.split_documents(data)

        # Ensure all documents are valid
        if not all(isinstance(doc, Document) for doc in docs):
            raise ValueError("Invalid document format. Ensure 'docs' is a list of Document objects.")

        # Generate embeddings and create FAISS index
        embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="small")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save FAISS index
        vectorstore.save_local(faiss_store_path)
        main_placeholder.text("FAISS Index Saved Successfully ✅")

    except Exception as e:
        st.error(f"Error during processing: {e}")

# Query step
# Query step
query = main_placeholder.text_input("Question: ", key="main_query_input")

if query:
    try:
        # Check if FAISS index exists
        if not os.path.exists(faiss_store_path):
            raise FileNotFoundError("FAISS index not found. Please process URLs first.")

        # Load FAISS index with dangerous deserialization enabled
        vectorstore = FAISS.load_local(
            faiss_store_path,
            embeddings=CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="small"),
            allow_dangerous_deserialization=True
        )

        # Set up retrieval QA chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except EOFError:
        st.error("Error: FAISS index is empty or corrupted. Please reprocess the URLs.")
    except Exception as e:
        st.error(f"Error while querying: {e}")