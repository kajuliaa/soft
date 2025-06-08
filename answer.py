
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from collections import defaultdict
import os
from dotenv import load_dotenv

# Load API KEY
load_dotenv()
api_key= os.getenv('GEMINI_API_KEY')


embedding = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory="memory_db", embedding_function=embedding)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=api_key
)

def build_prompt(grouped_articles):
    full_prompt = "You are an expert in AI field. Below is a collection of news article and corresponding images. Answer user question in user-friendly way based on provided textual and visual information.\n\n"
    for i, (article_id, chunks) in enumerate(grouped_articles.items(), 1):
        image_url = chunks[0].metadata.get('image_url', 'N/A')
        content_snippets = "\n".join(f"- {doc.page_content.strip()}" for doc in chunks)
        full_prompt += f"""🖼️ Image URL: {image_url}\n📚 Information:\n{content_snippets}\n\n"""
    return full_prompt

def run_rag_pipeline(query: str):
    retrieved_docs = db.similarity_search(query, k=20)
    grouped_articles = defaultdict(list)
    for doc in retrieved_docs:
        article_id = doc.metadata['article_id']
        grouped_articles[article_id].append(doc)

    prompt = build_prompt(grouped_articles)
    final_prompt = f"{prompt}\n\nUser question: {query}"
    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content, grouped_articles

# evaluating of 
# probably make link clicable to whole article