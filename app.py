import streamlit as st
from dotenv import load_dotenv
import pickle
import time
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI  # Changed this import
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_groq import ChatGroq
import os

# Load environment variables from a .env file if present
load_dotenv()

# Configure Gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please check your .env file.")
    st.stop()

try:
    if not GROQ_API_KEY or not GROQ_API_KEY.startswith('gsk_'):
        st.error("Please provide a valid Groq API key starting with 'gsk_'")
        st.stop()
        
    # Use the correct model name for Groq
    MODEL_NAME = "mixtral-8x7b"  # Simplified model name
    
    # Test connection with a simple completion
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )
    
    # Test the connection with a simple completion
    test_response = llm.invoke("Hello")
    st.success(f"Successfully connected to Groq using model: {MODEL_NAME}")
    
except Exception as e:
    st.error(f"Error configuring Groq: {str(e)}")
    st.error("Please verify your API key and internet connection")
    st.stop()

# Sidebar contents
with st.sidebar:
    st.title('Common questions asked during pregnancy')
  
    # Path to your image file
    image_path = "image.jpg"

    # Display the image
    st.image(image_path, caption='Helping Pregnant women', use_container_width=True)
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Llama 2](https://ai.meta.com/llama/) via Groq
                
    Built by [Nishu Pandey](https://www.linkedin.com/in/nishupandey085/)
    ''')
    add_vertical_space(5)

# Add this after loading the API key
import requests

def test_groq_connection():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    try:
        response = requests.get("https://api.groq.com/v1/models", headers=headers)
        if response.status_code == 200:
            st.success("Successfully connected to Groq API")
            available_models = response.json()
            st.write("Available models:", [model["id"] for model in available_models["data"]])
        else:
            st.error(f"Failed to connect to Groq API: {response.status_code}")
    except Exception as e:
        st.error(f"Error testing Groq connection: {str(e)}")

# Add this after API key verification
test_groq_connection()

def main():
    st.header("Pregnancy Chatbot💬")

    # Get the file path from the user
    try:

        file_path = "Common-Questions-in-Pregnancy-pdf.pdf"

        # Check if a file path is provided
        if file_path:
            # Open the PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)

                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=250,
                    chunk_overlap=25,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                # Extracting store name from the file path
                store_name = os.path.splitext(os.path.basename(file_path))[0]
                # st.write(f'{store_name}')

                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        VectorStore = pickle.load(f)
                else:
                    # Modified embedding creation
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    model_kwargs = {'device': 'cpu'}
                    encode_kwargs = {'normalize_embeddings': True}

                    # Update the embeddings initialization
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )

                    try:
                        st.write("Creating vector store...")
                        VectorStore = FAISS.from_texts(
                            texts=[str(t) for t in chunks],  # Convert chunks to strings
                            embedding=embeddings
                        )
                        st.write("Vector store created successfully")
                    
                        with open(f"{store_name}.pkl", "wb") as f:
                            pickle.dump(VectorStore, f)
                            st.write("Saved to pickle file")
                    except Exception as e:
                        st.write(f"Error creating vector store: {str(e)}")
                        raise e
                    
            # Initialize the chat messages history
            if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"role": "assistant", "content": "Hello. How can I help?"}]

            # Prompt for user input and save
            if prompt := st.chat_input():
                st.session_state.messages.append({"role": "user", "content": prompt})
                docs = VectorStore.similarity_search(query=prompt, k=3)

            # display the existing chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # If last message is not from astext_splittersistant, we need to generate a new response
            if st.session_state.messages[-1]["role"] != "assistant":
                # Call LLM
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            llm = ChatGroq(
                                temperature=0.7,
                                groq_api_key=GROQ_API_KEY,
                                model_name=MODEL_NAME,
                                max_tokens=1024
                            )

                            context = "\n".join([doc.page_content for doc in docs[:2]])  # Only use top 2 results
                            context = context[:1000] if len(context) > 1000 else context  # Limit context size
                            
                            # Create the full prompt with context
                            full_prompt = f"""Based on this context about pregnancy, Provide a brief answer:

                            Context: {context}

                            Question: {prompt}

                            Answer (be consise): """

                            try:
                                response = llm.invoke(
                                    [HumanMessage(content=full_prompt)],
                                ).content
                                st.write(response)
                                message = {"role": "assistant", "content": response}
                                st.session_state.messages.append(message)
                            
                            except Exception as e:
                                if "429" in str(e):
                                    st.error("API rate limit reached. Please wait a moment and try again.")
                                else:
                                    st.error(f"Error: {str(e)}")
                            
                        except Exception as e:
                            st.error("Unable to process request. Please try again in a moment.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")

if __name__ == '__main__':
    main()