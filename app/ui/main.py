import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="AI News Agent",
    page_icon="📰",
    layout="wide"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")

def main():
    st.title("📰 AI News Agent")
    st.markdown("""
    Welcome to the AI News Agent! Ask questions about current events and get intelligent summaries
    of relevant news articles.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        max_results = st.slider("Maximum results", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI-powered news agent uses Gemini API and LangGraph to:
        - Search and analyze news articles
        - Generate intelligent summaries
        - Filter for relevance
        - Provide chain-of-thought reasoning
        """)
    
    # Main content
    query = st.text_input(
        "What would you like to know about?",
        placeholder="e.g., What are the latest developments in AI technology?"
    )
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching for relevant news..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/query",
                        json={"query": query, "max_results": max_results}
                    )
                    
                    if response.status_code == 200:
                        summaries = response.json()
                        
                        # Display results
                        for i, summary in enumerate(summaries, 1):
                            with st.container():
                                st.markdown(f"### {i}. {summary['title']}")
                                st.markdown(f"**Source:** {summary['source']}")
                                st.markdown("---")
                                st.markdown(summary['summary'])
                                
                                if summary.get('relevance_score'):
                                    st.markdown(
                                        f"*Relevance Score: {summary['relevance_score']:.2f}*"
                                    )
                                st.markdown("---")
                    else:
                        st.error("Error processing your query. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query to search for news.")

if __name__ == "__main__":
    main() 