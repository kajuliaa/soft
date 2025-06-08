# Retrieval-Augmented Question Answering System
This application retrieves latest information from `The Batch` and answer users question in field of AI News.

## Repository structure
- `ingestion.py` - Data collection, preprocessing, saving to database.
-  `answer.py` - Retrieval, prompt building, and LLM response generation.
-  `app.py` - UI
-  `evaluation.py` - script to access quality of the rag system

### `ingestion.py` - Article Scraper
- Scrapes with Chronium via Playwright news articles and metadata such as image link and article title
- splits text into chuncks for improved semantic retrieval and not to execeed LLm access
- embeds chuncks with `mxbai-embed-large` from Ollama, Generates a UUID for each article to keep chunks grouped correctly
- stores chuncks in ChromaDB
- Automatically follows pagination via “Older Posts” link for full data coverage

Before using the application. Ingestion.py script need to be run to collect the data.

### `answer.py`
- uses same embedding model from ollama for question of the user to be compatible with the articles embeddings
- rag pipeline
- - accept a user quera, transforms it in embedding
  - finds top 20 relevant chuncks from db (why 20)
  - groups chucnks by article id, to keep chuncks from one article together
  - combines text chunks, prompt and image links in single prompt
  - Image URLs are included in the prompt, enabling the LLM to consider visual information
  - Sends the constructed prompt with images to Gemini Flash 2 via ChatGoogleGenerativeAI

### `app.py`
  - provides interactive interface
  -   accepts user queries and triggers the RAG pipeline
  -   displays gemini response + relevant articles with corresponding images
       
  -Due to not all images are extracted correct, skips not available images

# Instructions to run
get api key for Gemini https://aistudio.google.com/apikey
create .env and paste your api key there

```bash
GEMINI_API_KEY=your_api_key_here
```

Install Ollama
install embedding model

1. Ingest the articles
   ```bash
python ingestion.py
```
2. run the streamlit
```bash
streamlit run app.py
```
