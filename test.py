from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model='mxbai-embed-large')
db = Chroma(persist_directory="memory_db", embedding_function=embedding)

docs = db.similarity_search("AI trends", k=5)
for doc in docs:
    print("📝 Text Snippet:", doc.page_content[:200], "...")
    print("🆔 Article ID:", doc.metadata['article_id'])
    print("📰 Title:", doc.metadata['title'])
    print("🔗 URL:", doc.metadata['url'])
    print("🖼️ Image URL:", doc.metadata['image_url'])
    print("-" * 80)
