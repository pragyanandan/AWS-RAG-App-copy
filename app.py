import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Access AWS credentials from environment variables  [To RUN at LOCAL]
#aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
#aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# Access AWS credentials from environment variables  [To RUN at streamlit]
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]

# Initialize boto3 client with error handling
try:
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
except Exception as e:
    logging.error(f"Error initializing boto3 client: {e}")
    st.error("AWS Bedrock client initialization failed. Check your credentials and configuration.")

# Initialize Bedrock Embeddings with logging
try:
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
except Exception as e:
    logging.error(f"Error initializing Bedrock Embeddings: {e}")
    st.error("Failed to initialize Bedrock Embeddings. Check your model configuration.")

# Define the prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

# Create the PromptTemplate object
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Data ingestion function
def data_ingestion():
    current_directory = os.getcwd()
    data_folder = os.path.join(current_directory, "data")
    logging.info(f"Loading PDFs from: {data_folder}")
    
    loader = PyPDFDirectoryLoader(data_folder)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    return docs

# Vector store function with improved logging and error handling
def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local("faiss_index")
        logging.info("Vector store created and saved locally.")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("Failed to create vector store. Check the documents or embedding configuration.")

# Function to get the Claude model with logging
def get_claude_llm():
    try:
        llm = Bedrock(
            model_id="anthropic.claude-v2:1",
            client=bedrock,
            model_kwargs={'max_tokens_to_sample': 512}
        )
        logging.info("Initialized Claude model.")
        return llm
    except Exception as e:
        logging.error(f"Error initializing Claude LLM: {e}")
        st.error("Failed to initialize Claude LLM.")

# Function to get the Llama2 model with logging
def get_llama2_llm():
    try:
        llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
        logging.info("Initialized Llama2 model.")
        return llm
    except Exception as e:
        logging.error(f"Error initializing Llama2 LLM: {e}")
        st.error("Failed to initialize Llama2 LLM.")

# Function to get LLM responses with the vector store and query
def get_response_llm(llm, vectorstore_faiss, query):
    # Create a RetrievalQA instance using the provided LLM and vector store
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Adjust chain type based on your use case
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,  # This ensures both result and source documents are returned
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Run the query through the QA chain and get the full response
    response = qa({"query": query})

    # Extract and return only the 'result' key from the response
    answer = response.get('result', "No answer found")
    return answer


# Main function
def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                try:
                    docs = data_ingestion()
                    get_vector_store(docs)
                    st.success("Vector store updated.")
                except Exception as e:
                    logging.error(f"Error updating vector store: {e}")
                    st.error("Failed to update vector store.")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_claude_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Done.")
            except Exception as e:
                logging.error(f"Error generating Claude output: {e}")
                st.error("Failed to generate output using Claude LLM.")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama2_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Done.")
            except Exception as e:
                logging.error(f"Error generating Llama2 output: {e}")
                st.error("Failed to generate output using Llama2 LLM.")

# Entry point
if __name__ == "__main__":
    main()
