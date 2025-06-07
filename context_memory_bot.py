import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# === Streamlit config must be first ===
st.set_page_config(page_title="🎓 RAG Memory Chatbot", layout="centered")

# === Load environment variables ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === Chat history directory and session ===
CHAT_DIR = "chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)
SESSION_PATH = os.path.join(CHAT_DIR, "session.json")

# === Load and process documents for RAG ===
@st.cache_resource
def load_documents():
    loader = TextLoader("context.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

@st.cache_resource
def create_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(_docs, embeddings)

documents = load_documents()
vectorstore = create_vectorstore(documents)

# === Memory and prompt ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Always give short, clear, and precise answers unless the user asks for more detail.
If the user says \"tell me more\" or \"explain in detail\", then you can expand. Otherwise, be brief.

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}
""")

# === Load previous chat ===
if os.path.exists(SESSION_PATH):
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        chat_log = json.load(f)
else:
    chat_log = []

for msg in chat_log:
    memory.chat_memory.add_user_message(msg["user"])
    memory.chat_memory.add_ai_message(msg["bot"])

# === RAG QA Chain ===
llm = ChatOpenAI(model="mistralai/mixtral-8x7b-instruct", temperature=0.3)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    output_key="answer"
)

# === UI Styling ===
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #1e1e1e, #2c2c2c, #3a3a3a);
        color: #fff;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.02);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
    }
    .block-container {
        max-width: 850px;
        margin: auto;
    }
    .stChatMessageContent {
        font-size: 1rem;
        padding: 1rem;
        background-color: #2b2b2b;
        color: #f1f1f1;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar for saved chats ===
st.sidebar.title("📁 Saved Chats")
saved_files = [f for f in os.listdir(CHAT_DIR) if f.endswith(".json") and f != "session.json"]
selected_file = st.sidebar.selectbox("Choose a saved chat:", ["None"] + saved_files)

if selected_file != "None" and selected_file:
    with open(os.path.join(CHAT_DIR, selected_file), "r", encoding="utf-8") as f:
        saved_chat = json.load(f)
    st.sidebar.success(f"Loaded {selected_file}")
    for msg in saved_chat:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["bot"])
else:
    st.title("🤖 RAG Memory Chatbot")
    st.caption("Chat with memory and knowledge base")

    for msg in chat_log:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["bot"])

    user_input = st.chat_input("Type your question...")
    if user_input:
        result = rag_chain({"question": user_input})
        response = result["answer"]

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

        chat_log.append({"user": user_input, "bot": response})
        with open(SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump(chat_log, f, indent=2, ensure_ascii=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save Chat"):
            save_name = st.text_input("Enter filename to save:", "saved_chat.json")
            if save_name:
                with open(os.path.join(CHAT_DIR, save_name), "w", encoding="utf-8") as f:
                    json.dump(chat_log, f, indent=2, ensure_ascii=False)
                st.success(f"Chat saved as {save_name}")

    with col2:
        if st.button("🆕 New Chat"):
            if os.path.exists(SESSION_PATH):
                os.remove(SESSION_PATH)
            st.rerun()

# === Optional loaders for different file types ===
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredSQLLoader
# def load_pdf(path): return PyPDFLoader(path).load()
# def load_sql(path): return UnstructuredSQLLoader(path).load()
# Replace loader above if needed for PDF/SQL/etc.
