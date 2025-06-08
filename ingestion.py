import uuid
import asyncio
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from playwright.async_api import async_playwright


class ArticleIngestor:
    def __init__(self, embedding_model='mxbai-embed-large', persist_dir='memory_db'):
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        self.base_url = "https://www.deeplearning.ai"
        self.all_chunks = []

    async def scrape_articles(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            current_url = f"{self.base_url}/the-batch/"

            while True:
                print(f"Loading page: {current_url}")
                await page.goto(current_url, timeout=60000)
                await page.wait_for_load_state("networkidle")

                article_els = page.locator("article")
                count = await article_els.count()
                print(f"Found {count} articles on this page.")

                for i in range(count):
                    article_doc = await self.extract_article(article_els.nth(i), browser)
                    self.all_chunks.extend(article_doc)

                older_posts = page.locator("a:has-text('Older Posts')")
                if await older_posts.count() > 0:
                    current_url = await older_posts.first.get_attribute("href")
                    print(f"Going to next page: {current_url}")
                else:
                    print("No 'Older Posts' link found. Scraping complete.")
                    break

            await browser.close()

    async def extract_article(self, article_el, browser):
        title = await article_el.locator("h2").inner_text()
        anchors = article_el.locator("a")
        counta = await anchors.count()
        if counta > 1:
            link = await anchors.nth(1).get_attribute("href")
        else:
            link = await anchors.first.get_attribute("href")

        if not link.startswith("http"):
            link = self.base_url + link

        # Extract article content
        article_page = await browser.new_page()
        await article_page.goto(link)
        await article_page.wait_for_load_state("networkidle")

        try:
            paragraphs = await article_page.locator("div.prose--styled p").all_inner_texts()
            content = "\n".join(paragraphs).strip()
        except Exception:
            content = ""

        await article_page.close()

        # Extract image url
        img_el = article_el.locator("img")
        img_url = await img_el.get_attribute("src")

        chunks = self.text_splitter.split_text(content)
        article_id = str(uuid.uuid4())

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "article_id": article_id,
                    "title": title.strip(),
                    "url": link,
                    "image_url": img_url or "N/A"
                }
            )
            for chunk in chunks
        ]

        return documents

    async def ingest(self):
        await self.scrape_articles()
        print(f"Storing {len(self.all_chunks)} chunks in ChromaDB...")
        db = Chroma.from_documents(self.all_chunks, self.embedding, persist_directory=self.persist_dir)
        db.persist()
        print("✅ Ingestion completed and saved to Chroma.")
        return self.all_chunks


async def main():
    ingestor = ArticleIngestor()
    await ingestor.ingest()


if __name__ == "__main__":
    asyncio.run(main())
