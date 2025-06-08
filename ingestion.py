from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from playwright.async_api import async_playwright
from langchain_core.documents import Document
import uuid
import asyncio
from langchain_community.vectorstores import Chroma

embedding = OllamaEmbeddings(model='mxbai-embed-large')
memory_db_path = 'memory_db' 

# splits text into smaller chuncks, respecting sentence oaragraph structure
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

async def extract_and_embed_articles():
    #starts a chronium browser in headless mode (no GUI)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Beggin of scraping from the batch
        base_url = "https://www.deeplearning.ai"
        current_url = base_url + "/the-batch/"
        all_chunks = []

        while True:
            print(f"Loading page: {current_url}")
            await page.goto(current_url, timeout=60000)
            await page.wait_for_load_state("networkidle")

            # scrape all articles in the czrrent page
            article_els = page.locator("article")
            count = await article_els.count()
            print(f"Found {count} articles on this page.")

            #get the title of each article and link
            for i in range(count):
                el = article_els.nth(i)
                title = await el.locator("h2").inner_text()

                anchors = el.locator("a")
                counta = await anchors.count()

                #use the second link if available, if not then first. Was throughwing an error
                if counta > 1:
                    link = await anchors.nth(1).get_attribute("href")
                else:
                    link = await anchors.first.get_attribute("href")
                if not link.startswith("http"):
                    link = base_url + link


                # open article page and extract text from p inside div
                article_page = await browser.new_page()
                await article_page.goto(link)
                await article_page.wait_for_load_state("networkidle")
                try:
                    paragraphs = await article_page.locator("div.prose--styled p").all_inner_texts()
                    content = "\n".join(paragraphs).strip()
                except:
                    content = ""
                await article_page.close()

                # get this article image
                img_el = el.locator("img")
                img_url = await img_el.get_attribute("src")
                if img_url and not img_url.startswith("http"):
                    img_url = base_url + img_url

                # split and store chuncks and generate unqie id
                chunks = text_splitter.split_text(content)
                article_id = str(uuid.uuid4())
                # wraps each chunk into a LangChain Document with metadata (titke, URL, Image, ID)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "article_id": article_id,
                            "title": title.strip(),
                            "url": link,
                            "image_url": img_url
                        }
                    )
                    all_chunks.append(doc)
            # go to the next page
            older_posts = page.locator("a:has-text('Older Posts')")
            if await older_posts.count() > 0:
                older_posts_link = await older_posts.first.get_attribute("href")
                if older_posts_link:
                    if not older_posts_link.startswith("http"):
                        current_url = base_url + older_posts_link
                    else:
                        current_url = older_posts_link
                        print(f"Going to next page: {current_url}")
                else:
                    print("Found 'Older Posts' link, but no href. Stopping.")
                    break
            else:
                print("No 'Older Posts' link found. Scraping complete.")
                break

        await browser.close()

        print(f"Storing {len(all_chunks)} chunks in Chroma...")
        #embeds all collected document chunks and store them in chromadb
        db = Chroma.from_documents(all_chunks, embedding, persist_directory=memory_db_path)
        db.persist()
        print("✅ Ingestion completed and saved to Chroma.")

        return all_chunks

async def main():
    await extract_and_embed_articles()
if __name__ == "__main__":
    asyncio.run(main())