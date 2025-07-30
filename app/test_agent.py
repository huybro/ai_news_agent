import os
import json
import traceback
from dotenv import load_dotenv
import requests

# It's better to handle potential import errors, especially for relative imports.
try:
    # Assuming your script is run from the root of the 'ai_news_agent' directory
    from app.agents.news_agent import NewsAgent
except ImportError:
    print("Error: Could not import NewsAgent. Make sure you are running this script from the project's root directory.")
    # Add the project root to the path if necessary
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.agents.news_agent import NewsAgent


def test_eventregistry_api():
    """Test the EventRegistry API directly to ensure the key and connection work."""
    print("\n=== Testing EventRegistry API Connection ===")
    
    # FIX: Get the API key directly. load_dotenv() is now called in the main block.
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("Error: NEWS_API_KEY not found. Ensure it's in your .env file.")
        return

    payload = {
        "action": "getArticles",
        "keyword": "Tesla stock performance",
        "articlesCount": 5,
        "apiKey": api_key,
        "resultType": "articles",
    }
    
    try:
        response = requests.post(
            "https://eventregistry.org/api/v1/article/getArticles",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # This will raise an exception for bad status codes

        data = response.json()
        articles = data.get("articles", {}).get("results", [])
        print(f"Success: API returned {len(articles)} articles.")

    except requests.exceptions.HTTPError as e:
        print(f"API HTTP Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

def test_news_agent():
    """Test the complete NewsAgent workflow from query to summaries."""
    print("\n=== Testing Full News Agent Workflow ===")
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if not gemini_api_key or not news_api_key:
        print("Error: Ensure both GEMINI_API_KEY and NEWS_API_KEY are set.")
        return
    
    try:
        # FIX: The NewsAgent now requires both keys in its constructor.
        print("Initializing News Agent...")
        agent = NewsAgent(gemini_api_key=gemini_api_key, news_api_key=news_api_key)
        
        test_query = "visa in the usa"
        print(f"Processing query: '{test_query}'")
        
        # FIX: The method was renamed from 'process_query' to 'run'.
        result = agent.run(test_query)
        
        print("\n--- AGENT RUN COMPLETE ---")
        if result.get("error"):
            print(f"An error occurred during the agent run: {result['error']}")
            return
            
        # FIX: The result structure is now simpler. We iterate through the list of summaries.
        summaries = result.get("summaries", [])
        articles = result.get("articles", [])
        
        print(f"\nQuery: {result.get('query')}")
        print(f"Found {len(articles)} relevant articles and generated {len(summaries)} summaries.")
        
        if summaries:
            print("\n--- Generated Summaries ---")
            for i, summary in enumerate(summaries):
                print(f"{i+1}. {summary}")
        else:
            print("\nNo summaries were generated. There may have been no relevant articles found.")
            
    except Exception as e:
        print(f"A critical error occurred while testing the agent: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # FIX: Load .env file once at the very beginning.
    # This ensures environment variables are available for all functions.
    load_dotenv()
    
    # Run the direct API test first to check credentials.
    test_eventregistry_api()
    
    # Run the full agent test.
    test_news_agent()
