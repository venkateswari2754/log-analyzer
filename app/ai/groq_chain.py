from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

from app.ai.embeddings import get_embeddings_model
from app.cache.vector_cache import save_vector_store, load_vector_store

def setup_ai_chain(analyzer, groq_api_key: str, model_name: str = "llama3-8b-8192") -> bool:
    try:
        if analyzer.log_data is None:
            raise ValueError("Log data not found")

        # Prepare documents from log lines
        documents = []
        for row in analyzer.log_data.to_dict(orient="records"):
            content = row["raw_line"]
            metadata = {
                "timestamp": row.get("timestamp"),
                "level": row.get("level"),
                "component": row.get("component"),
                "error_type": row.get("error_type")
            }
            documents.append(Document(page_content=content, metadata=metadata))

        # Load cached vector store if available
        vector_store = load_vector_store(analyzer.log_hash)
        if not vector_store:
            embeddings = get_embeddings_model()
            vector_store = FAISS.from_documents(documents, embeddings)
            save_vector_store(analyzer.log_hash, vector_store)
        analyzer.vector_store = vector_store

        # Setup LLM
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        analyzer.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        return True
    except Exception as e:
        print(f"[setup_ai_chain error]: {str(e)}")
        return False

def query_logs(analyzer, query: str) -> str:
    if not analyzer.qa_chain:
        return "❌ AI not initialized yet."
    try:
        result = analyzer.qa_chain({"query": query})
        return result['result']
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_similar_logs(analyzer, search_query: str, k: int = 5) -> list:
    if not analyzer.vector_store:
        return []

    try:
        docs = analyzer.vector_store.similarity_search(search_query, k=k)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return results
    except Exception as e:
        print(f"[get_similar_logs error]: {str(e)}")
        return []