# PregnaCare AI Assistant

An intelligent AI-powered chatbot that provides accurate pregnancy-related information using Groq's Mixtral-8x7b LLM.

## Features
- Real-time chat interface
- PDF document processing and semantic search
- Context-aware responses using RAG architecture
- Error handling and API connection validation

## Tech Stack
- Python
- Streamlit
- LangChain
- Groq API
- FAISS Vector Store
- HuggingFace Embeddings
- PyPDF2

## Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Create a `.env` file and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the app: `streamlit run app.py`

## Usage
The chatbot will process the pregnancy-related PDF and create a knowledge base. Users can then ask questions about pregnancy, and the bot will provide context-aware responses.

## Created by
[Your Name](your-linkedin-url)
```

```text requirements.txt
streamlit
python-dotenv
PyPDF2
streamlit-extras
langchain
faiss-cpu
transformers
sentence-transformers
torch
langchain-groq
requests
```

```text .gitignore
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual Environment
venv/
env/

# IDE
.vscode/
.idea/

# Pickle files
*.pkl

# Distribution
dist/
build/
*.egg-info/
```
