import streamlit as st
from answer import run_rag_pipeline



def main():
    st.set_page_config(page_title="📰 Recent in the AI", layout="wide")

    st.title("🔍 Recent in the AI")
    st.markdown("Ask a question about AI to get information about recent researches and news!")

    user_query = st.text_input("Enter your question", value="How is generative AI impacting education?")
    if st.button("Submit") and user_query:
        with st.spinner("Searching and analyzing articles..."):
            response, articles = run_rag_pipeline(user_query)

        st.subheader("🤖 Answer")
        st.markdown(response)

        st.subheader("📄 Related Articles")
        for i, (article_id, chunks) in enumerate(articles.items(), 1):
            title = chunks[0].metadata.get('title', 'Untitled')
            url = chunks[0].metadata.get('url', '#')
            st.markdown(f"**{i}. [{title}]({url})**")

            img_url = chunks[0].metadata.get("image_url", "")
            if img_url == 'https://www.deeplearning.aidata:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7':
                continue
            else:
                st.image(img_url, width=500)


if __name__ == "__main__":
    main()
