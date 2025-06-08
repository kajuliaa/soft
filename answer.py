import os
from collections import defaultdict
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')


class RAGPipeline:
    def __init__(self, embedding_model="mxbai-embed-large", persist_dir="memory_db"):
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embedding)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=API_KEY
        )

    def build_prompt(self, grouped_articles: dict) -> str:
        prompt = (
            "You are an expert in AI field. Below is a collection of news articles and corresponding images. "
            "Answer user question in a user-friendly way based on the provided textual and visual information.\n\n"
        )
        for i, (article_id, chunks) in enumerate(grouped_articles.items(), 1):
            image_url = chunks[0].metadata.get('image_url', 'N/A')
            content_snippets = "\n".join(f"- {doc.page_content.strip()}" for doc in chunks)
            prompt += f"🖼️ Image URL: {image_url}\n📚 Information:\n{content_snippets}\n\n"
        return prompt

    def run(self, query: str):
        retrieved_docs = self.db.similarity_search(query, k=20)
        grouped_articles = defaultdict(list)
        for doc in retrieved_docs:
            grouped_articles[doc.metadata['article_id']].append(doc)

        prompt = self.build_prompt(grouped_articles)
        final_prompt = f"{prompt}\n\nUser question: {query}"
        response = self.llm.invoke([HumanMessage(content=final_prompt)])
        return response.content, grouped_articles


# Singleton instance for app use
rag_pipeline = RAGPipeline()


def run_rag_pipeline(query: str):
    return rag_pipeline.run(query)
