
import streamlit as st
from answer import run_rag_pipeline
import base64
from io import BytesIO
import streamlit as st
from PIL import Image
import requests

def is_valid_image_url(url: str) -> bool:
    if not url or url == "N/A":
        return False
    if url.startswith("data:image") or "base64" in url:
        return False
    if not url.startswith("http"):
        return False
    if not any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
        return False
    return True



st.set_page_config(page_title="📰 Recent in the AI", layout="wide")

st.title("🔍 Recent in the AI")
st.markdown("Ask a question about AI to get an information about recent researches and news!")

user_query = st.text_input("Enter your question", value="How is generative AI impacting education?")
if st.button("Submit") and user_query:
    with st.spinner("Searching and analyzing articles..."):
        response, articles = run_rag_pipeline(user_query)

    st.subheader("🤖 Answer")
    st.markdown(response)



    st.subheader("📄 Related Articles")
    for i, (article_id, chunks) in enumerate(articles.items(), 1):
        title = chunks[0].metadata.get('title', 'Untitled')
        st.markdown(f"**{i}. {title}**")

        img_url = chunks[0].metadata.get("image_url", "")
        #st.text(f"[DEBUG] Image URL: {img_url}")
    
        if img_url == 'https://www.deeplearning.aidata:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7':
            continue
        else:
            st.image(img_url, width=500)


        
        
