import os
import time
import json
from answer import run_rag_pipeline, rag_pipeline

RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_query(query: str, k: int = 20):
    result = {
        "query": query,
        "retrieval": [],
        "gemini_response": "",
        "duration": 0
    }

    docs = rag_pipeline.db.similarity_search(query, k=k)
    for doc in docs:
        result["retrieval"].append({
            "title": doc.metadata.get("title", "N/A"),
            "url": doc.metadata.get("url", "N/A"),
            "image_url": doc.metadata.get("image_url", "N/A"),
            "snippet": doc.page_content
        })


    start = time.time()
    response, articles = run_rag_pipeline(query)
    result["duration"] = round(time.time() - start, 2)
    result["gemini_response"] = response

    return result

def save_json_result(query_result: dict):
    filename_safe_query = query_result['query'].replace(" ", "_").replace("?", "").lower()
    filepath = os.path.join(RESULTS_DIR, f"{filename_safe_query}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(query_result, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON: {filepath}")


if __name__ == "__main__":
    test_queries = [
        "How is generative AI used in education?",
        "Recent advancements in robotics",
        "AI in healthcare",
    ]

    for query in test_queries:
        print(f"\n🔍 Evaluating: '{query}'")
        result = evaluate_query(query)
        save_json_result(result)
