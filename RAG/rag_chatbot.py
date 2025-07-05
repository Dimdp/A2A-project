from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import glob
import os

llm = OllamaLLM(model="mistral")
embed_model = OllamaEmbeddings(model="mistral")

class QAState(TypedDict):
    question: str
    vectordb: Chroma
    answer: str
    chat_history: List[str]  # Add chat history to state

def load_and_chunk(folder_path):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {folder_path}")
        return []
    
    for path in pdf_paths:
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    
    return all_chunks

def store_chunks(chunks):
    vectordb = Chroma.from_documents(chunks,
                                   embedding=embed_model,
                                   persist_directory="./db")
    vectordb.persist()
    return vectordb

def qa_node(state):
    retriever = state["vectordb"].as_retriever()
    
    # Create enhanced question with chat history context
    chat_history = state.get("chat_history", [])
    
    # Build context-aware question
    if chat_history:
        context = "\n".join(chat_history[-6:])  # Last 3 exchanges (6 messages)
        enhanced_question = f"""
        Previous conversation:
        {context}
        
        Current question: {state['question']}
        
        Please answer the current question considering the conversation history above.
        """
    else:
        enhanced_question = state['question']
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.invoke(enhanced_question)
    
    if isinstance(answer, dict) and "result" in answer:
        answer_text = answer["result"]
    else:
        answer_text = str(answer)
    
    state['answer'] = answer_text
    
    # Update chat history in the state
    updated_history = state.get("chat_history", []).copy()
    updated_history.append(f"Human: {state['question']}")
    updated_history.append(f"AI: {answer_text}")
    
    # Keep only last 10 exchanges (20 messages)
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]
    
    state['chat_history'] = updated_history
    return state

# Initialize graph
graph = StateGraph(QAState)
graph.add_node("qa", qa_node)
graph.add_edge(START, "qa")
graph.add_edge("qa", END)
qa_graph = graph.compile()

# Load or create vector database
if os.path.exists("./db"):
    vectordb = Chroma(persist_directory="./db", embedding_function=embed_model)
else:
    pdf_path = input("Enter the path to your PDF files: ")
    chunks = load_and_chunk(pdf_path)
    vectordb = store_chunks(chunks)

# Initialize memory - this will be passed through state
initial_chat_history = []

# ====== Interactive QA Loop with Memory ======
print('\n📘 Ask questions about your documents.')
print('💡 Type (quit, exit, q or clear) to terminate the chat ...')
print('🧠 Memory enabled - I can remember our conversation!\n')

while True:
    user_input = input("🧑‍💻 You: ")
    if user_input.lower() in ["quit", "exit", "q", "clear"]:
        print("👋 Goodbye!")
        break
    
    # Skip empty inputs
    if not user_input.strip():
        continue
    
    state = {
        "question": user_input, 
        "vectordb": vectordb,
        "chat_history": initial_chat_history.copy()
    }
    
    result = qa_graph.invoke(state)
    answer = result["answer"]
    
    # Update the persistent chat history with the result from state
    initial_chat_history = result["chat_history"]
    
    print("🤖 Answer:", answer)
    print()  # Add spacing for readability