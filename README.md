# Retrieval-Augmented Question Answering System
This project implements a Retrieval-Augmented Generation system that answers user queries based on recent AI news articles scraped from The Batch. 

DEMO: https://streamable.com/sci6di
## 🛠️ Technology Stack

|                | Tool/Model                        | Reasoning                                                                 |
|--------------------|-----------------------------------|---------------------------------------------------------------------------|
| **Embedding Model**| `mxbai-embed-large` (via Ollama)  | Open-source, locally runnable, high-quality text embedding for semantic retrieval. |
| **Vector Database**| ChromaDB                          | Lightweight, persistent vector store with metadata filtering capabilities. |
| **Web Scraping**   | Playwright                        | Handles dynamic content and JavaScript-rendered pages; supports full browser control. |
| **LLM (Generation)**| Gemini Flash 2 via `google.generativeai` | Fast, multimodal (text + image), cost-effective, and good for real-time applications. |
| **Frontend UI**    | Streamlit                         | Simple, fast web app development; ideal for prototyping ML applications. |


## Instructions to run
1. Get Gemini API Key:
   [Get from Google AI Studio](https://aistudio.google.com/apikey)

2. Create `.env` file and paste your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
3. Install [Ollama](https://ollama.com/)  and load `mxbai-embed-large` model.
4. Create enviroment and install dependencies:
   - `conda`
    ```bash
   conda create -n ai-rag-env python=3.11.3
   conda activate ai-rag-env
   pip install -r requirements.txt
    ```
   - `venv` (version for Linux/macOS)
   ```bash
   python -m venv ai-rag-env
   source ai-rag-env/bin/activate
   pip install -r requirements.txt
   ```
5. Install Playwright:
```bash
playwright install
```
6. Scrape and ingest the articles:
```bash
python ingestion.py
```
7. Run the Streamlit app:
```bash
streamlit run app.py
```

## Repository structure
- `ingestion.py` - Scrapes news articles, processes and embeds content, and stores it in ChromaDB
-  `answer.py` - Handles retrieval, prompt construction, and LLM response generation
-  `app.py` - Streamlit UI for interacting with the RAG system
-  `evaluation.py` - Evaluates system performance (retrieval + generation)


### `ingestion.py` - Article Ingestion
- Uses Chromium via Playwright to scrape news articles and metadata (image URL, article title)
- Splits article text into chunks for improved semantic retrieval and to stay within LLM context limits
- Embeds chunks using `mxbai-embed-large` via Ollama and assigns a UUID to group chunks per article
- Stores chunks in ChromaDB
- Automatically follows pagination via “Older Posts” link for full data coverage
  #### Methods
  `scrape_articles()`
  - Iteratively loads pages and scrapes article elements
  - Follows "Older Posts" link to paginate
  - Calls `extract_article()` for each article
    
  `extract_article(article_el, browser)`
  - Parses title, article URL, and image URL
  - Loads full article page
  - Splits content into chunks
  - saves object with metadata
 
  `ingest()`
  - Triggers scraping
  - Embeds all chunks
  - Persists them into ChromaDB

Note: Run ingestion.py once before using the application to ingest the initial dataset.

### `answer.py`
- Uses the same embedding model (`mxbai-embed-large`) to ensure query compatibility
- Accepts a user query and converts it into an embedding
- Retrieves top 20 most similar chunks from the database
- Groups chunks by article ID to keep related content together
- Combines grouped chunks, image URLs, and text into a single prompt
- Sends the constructed prompt with images to Gemini Flash 2 via ChatGoogleGenerativeAI
    #### Methods
  `build_prompt(grouped_articles)`
  - Constructs a formatted prompt string with images and related content
    
  `run(query)`
  - Embeds user query
  - Performs `similarity_search(k=20)`
  - Groups retrieved chunks by `article_id`
  - Sends prompt to Gemini LLM
 
  `run_rag_pipeline(query)`
  - Running the RAG pipeline

### `app.py`
  - Provides an interactive web interface
  - Accepts user questions and triggers the RAG pipeline
  - Displays Gemini’s answer with relevant articles and image previews    
  - Skips unavailable or missing images to avoid rendering errors

### `evaluation.py`
- Generates a dictionary for future assessment with the following structure:
  ```python
   {
  "query": ...,
  "retrieval": [...],  # Retrieved article chunks
  "generative": ...,   # LLM response
  "generation_time": ...
   }

  ```
#### Potential Future Metrics
- Retrieval Quality
   - Precision@k, Recall@k
- Answer Quality 
   - ROUGE, BLEU, or BERTScore 






# Limitations
- Only the first image of each article is currently extracted. Some articles contain multiple images which should also be included in future updates
- Evaluation with other techniques can be performed
- Cloud deployment is currently not supported, as Ollama models are designed for local use and require a custom server setup that is not easily compatible with standard cloud platforms
