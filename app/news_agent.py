import os
import logging
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore


# --- Configuration ---
# Load environment variables at the start
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
NEWS_API_URL = "https://eventregistry.org/api/v1/article/getArticles"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# --- Tool Definitions ---
# (Your tool definitions remain unchanged)
@tool
def search_for_news(query: str) -> str:
    """
    Searches for recent news articles about a specific topic.
    Returns a list of articles including their title, URL, and content as a JSON string.
    """
    logging.info(f"Tool 'search_for_news' called with query: {query}")
    if not NEWS_API_KEY:
        return "Error: News API key is not configured."

    payload = {
        "action": "getArticles", "keyword": query, "articlesCount": 8,
        "articlesSortBy": "rel", "dataType": ["news", "pr"],
        "apiKey": NEWS_API_KEY, "includeArticleBody": True,
        "isDuplicateFilter": "skipDuplicates"
    }

    try:
        response = requests.post(NEWS_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        raw_articles = data.get("articles", {}).get("results", [])
        if not raw_articles:
            return "No articles found for that topic."
        return json.dumps([{"title": a.get("title"), "content": a.get("body"), "url": a.get("url")} for a in raw_articles])
    except Exception as e:
        logging.error(f"Error in search_for_news tool: {e}")
        return f"Error searching for news: {e}"

@tool
def summarize_all_and_synthesize(articles_json: str) -> str:
    """
    Receives a JSON string containing a LIST of articles, summarizes each one,
    and then creates a final synthesized overview of all of them.
    """
    logging.info("Tool 'summarize_all_and_synthesize' called.")
    try:
        articles = json.loads(articles_json)
        if not isinstance(articles, list) or not articles:
            return "Error: Invalid or empty list of articles provided."
    except json.JSONDecodeError:
        return "Error: The input must be a valid JSON string of articles."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)
    contents = [f"Article {i+1}:\nTitle: {a.get('title')}\nContent: {a.get('content')}" for i, a in enumerate(articles)]
    prompt = (
        "You are a world-class news analyst. Your task is to do two things:\n"
        "1. Provide a concise, 1-2 sentence summary for EACH of the following articles.\n"
        "2. After summarizing them all, write a final, 'Key Takeaways' paragraph that synthesizes the main themes.\n\n"
        f"Here are the articles:\n\n{'---'.join(contents)}\n\n"
        "Please format your response clearly with a 'Summaries' section and a 'Key Takeaways' section."
    )
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        logging.error(f"Error during summarization/synthesis: {e}")
        return f"Error processing articles: {e}"

@tool
def summarize_one_article(article_json: str) -> str:
    """
    Receives a JSON string for a single article and provides a detailed summary.
    """
    logging.info("Tool 'summarize_one_article' called.")
    try:
        article = json.loads(article_json)
        if not isinstance(article, dict):
            return "Error: Invalid article format."
    except json.JSONDecodeError:
        return "Error: The input must be a valid JSON string."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)
    prompt = (
        f"Please provide a detailed summary of the following news article:\n\n"
        f"Title: {article.get('title', 'N/A')}\n\n"
        f"Content: {article.get('content', 'N/A')}\n\n"
        "The summary should capture the key points, main arguments, and any important conclusions."
    )
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        logging.error(f"Error during single article summarization: {e}")
        return f"Error summarizing article: {e}"


def create_agent(checkpointer, store: BaseStore):
    """Creates and returns the conversational agent executor."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    tools = [search_for_news, summarize_all_and_synthesize, summarize_one_article]
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY, )

    system_message = """You are a helpful news assistant. Your goal is to provide users with accurate and concise news information based on their requests.

    You have access to the following tools:
    - `search_for_news`: Use this to find recent articles on a given topic.
    - `summarize_all_and_synthesize`: Use this after searching to provide a comprehensive overview of multiple articles.
    - `summarize_one_article`: Use this if the user asks for a summary of a specific article you have found.

    Here is how you should operate:
    1.  When the user asks a general question about a news topic (e.g., "What's the latest on AI regulations?"), you MUST first use the `search_for_news` tool with a relevant query.
    2.  After getting the search results, present the user with a list of the article titles and their URLs. Ask the user if they would like you to summarize the articles. If they say yes, then use the `summarize_all_and_synthesize` tool.
    3.  If the user asks for more detail on a single article, you can use the `summarize_one_article` tool.
    4.  Always be polite and clear in your responses. If you cannot find any news, inform the user gracefully.
    5.  Do not make up information. Stick to the content provided by your tools."""

    return create_react_agent(
        model,
        tools,
        checkpointer=checkpointer,
        store=store,
        prompt=system_message
    )