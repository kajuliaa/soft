# Retrieval-Augmented Question Answering System
This application retrieves latest information from `The Batch` and answer users question in field of AI News.

## Instructions to run
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

## Repository structure
- `ingestion.py` - Scrapes news articles, processes and embeds content, and stores it in ChromaDB
-  `answer.py` - Handles retrieval, prompt construction, and LLM response generation
-  `app.py` - Streamlit UI for interacting with the RAG system
-  `evaluation.py` - Evaluates system performance (retrieval + generation)


### `ingestion.py` - Article Ingestion
- Scrapes with Chronium via Playwright news articles and metadata such as image link and article title
- splits text into chuncks for improved semantic retrieval and not to execeed LLm access
- embeds chuncks with `mxbai-embed-large` from Ollama, Generates a UUID for each article to keep chunks grouped correctly
- stores chuncks in ChromaDB
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




Before using the application. Ingestion.py script need to be run to collect the data.

### `answer.py`
- uses same embedding model from ollama for question of the user to be compatible with the articles embeddings
- rag pipeline
- accept a user quera, transforms it in embedding
- finds top 20 relevant chuncks from db (why 20)
- groups chucnks by article id, to keep chuncks from one article together
- combines text chunks, prompt and image links in single prompt
- Image URLs are included in the prompt, enabling the LLM to consider visual information
- Sends the constructed prompt with images to Gemini Flash 2 via ChatGoogleGenerativeAI
    #### Methods
  `build_prompt(grouped_articles)`
  - Constructs a formatted prompt string
  - Image URLs + Related text content per article
    
  `run(query)`
  - Embeds user query
  - Performs `similarity_search(k=20)`
  - Groups retrieved chunks by `article_id`
  - Calls `build_prompt()`
  - Sends prompt to Gemini LLM
 
  `run_rag_pipeline(query)`
  - Running the RAG pipeline

### `app.py`
  - provides interactive interface
  -   accepts user queries and triggers the RAG pipeline
  -   displays gemini response + relevant articles with corresponding images
       
  -Due to not all images are extracted correct, skips not available images
### `evaluation.py`
- creates a dict to future assesment with the structure: query, retrieval(retrieved chuncks of articles), generative ( what LLM responsed), generation time.
#### Possible future improvements
- retrieval quality
   - Precision@k, Recall@k, MRR — How well the system ranks relevant articles
- Answer Quality 
   - ROUGE, BLEU, or BERTScore — Against gold-standard responses






## Future improvements
- evaluate Precision@k, Recall@k, MRR — to assess the ranking quality for retrieval quality
- ROUGE, BLEU, or BERTScore vs. reference answers for answer quality

# Improvements
- all images from article should be get, not only first one (saw already after ingestion, that article can contain multiple images)
