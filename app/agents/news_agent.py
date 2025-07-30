import os
import re
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable

import requests
import google.generativeai as genai
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
NEWS_API_URL = "https://eventregistry.org/api/v1/article/getArticles"
CONTENT_PREVIEW_LENGTH = 400

class NewsAgentState(BaseModel):
    """Defines the state for the news processing agent."""
    query: str
    articles: List[Dict[str, Any]] = []
    summaries: List[str] = []
    error: Optional[str] = None

# --- Graph Node Decorator ---
def graph_node(step_name: str) -> Callable:
    """A decorator to wrap graph nodes, handling state, logging, and errors."""
    def decorator(func: Callable[[Any, NewsAgentState], NewsAgentState]) -> Callable:
        def wrapper(instance, state_input: dict | NewsAgentState) -> dict:
            if isinstance(state_input, dict):
                state = NewsAgentState(**state_input)
            else:
                state = state_input
            
            logging.info(f"--- Entering Step: {step_name} ---")
            
            if state.error:
                logging.warning(f"Skipping step '{step_name}' due to previous error: {state.error}")
                return state.model_dump()
            
            try:
                updated_state = func(instance, state)
                logging.info(f"--- Successfully Completed Step: {step_name} ---")
                return updated_state.model_dump()
            except Exception as e:
                logging.error(f"Error in step '{step_name}': {e}", exc_info=True)
                state.error = f"Failed during '{step_name}': {str(e)}"
                return state.model_dump()
        return wrapper
    return decorator

class NewsAgent:
    def __init__(self, gemini_api_key: str, news_api_key: str):
        """Initializes the News Agent and its processing workflow."""
        if not gemini_api_key or not news_api_key:
            raise ValueError("Both Gemini and News API keys are required.")
            
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.news_api_key = news_api_key
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph processing workflow."""
        workflow = StateGraph(NewsAgentState)
        
        workflow.add_node("search", self._search_step)
        workflow.add_node("filter", self._filter_step)
        workflow.add_node("summarize", self._summarize_step)

        workflow.add_edge(START, "search")
        workflow.add_edge("search", "filter")
        workflow.add_edge("filter", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()

    def run(self, query: str) -> Dict[str, Any]:
        """Runs the agent to process a query and return news summaries."""
        logging.info(f"Starting news processing for query: '{query}'")
        initial_state = {"query": query}
        
        try:
            final_state = self.graph.invoke(initial_state)
            logging.info("Workflow completed.")
            return final_state
        except Exception as e:
            logging.critical(f"A critical error occurred during workflow execution: {e}")
            traceback.print_exc()
            return {
                "query": query, "articles": [], "summaries": [],
                "error": f"Workflow execution failed: {str(e)}"
            }

    def _create_search_payload(self, query: str) -> Dict[str, Any]:
        """Creates the JSON payload for the news API request."""
        return {
            "action": "getArticles", "keyword": query, "articlesCount": 10,
            "articlesSortBy": "rel", "dataType": ["news", "pr"],
            "forceMaxDataTimeWindow": 31, "resultType": "articles",
            "apiKey": self.news_api_key, "includeArticleBody": True,
            "includeArticleTitle": True, "includeArticleBasicInfo": True,
            "isDuplicateFilter": "skipDuplicates" # Use API to handle duplicates
        }

    @graph_node("search")
    def _search_step(self, state: NewsAgentState) -> NewsAgentState:
        """Searches for news articles using the EventRegistry API."""
        payload = self._create_search_payload(state.query)
        response = requests.post(NEWS_API_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        raw_articles = data.get("articles", {}).get("results", [])
        if not raw_articles:
            raise ValueError("No articles found in the API response.")

        state.articles = [{
            "title": article.get("title"), "content": article.get("body"),
            "source": article.get("source", {}).get("title", "Unknown"),
            "url": article.get("url"),
        } for article in raw_articles]
        
        logging.info(f"Found {len(state.articles)} unique articles.")
        return state

    @graph_node("filter")
    def _filter_step(self, state: NewsAgentState) -> NewsAgentState:
        """Filters articles for relevance in a single batch API call."""
        if not state.articles:
            logging.warning("No articles to filter.")
            return state

        # Create a single prompt with all article previews
        article_previews = []
        for i, article in enumerate(state.articles):
            preview = article.get('content', '')[:CONTENT_PREVIEW_LENGTH]
            article_previews.append(f"Article {i+1}:\nTitle: {article.get('title')}\nPreview: {preview}...")

        prompt = (
            f"The user is asking about: '{state.query}'.\n\n"
            "Below are several news articles. Identify which ones are relevant to the user's query.\n\n"
            f"{'---'.join(article_previews)}\n\n"
            "List the numbers of the relevant articles, separated by commas (e.g., '1, 3, 5')."
        )
        
        response = self.model.generate_content(prompt)
        
        # Parse the model's response to get the indices of relevant articles
        try:
            relevant_indices = {int(i.strip()) - 1 for i in response.text.split(',') if i.strip().isdigit()}
            state.articles = [state.articles[i] for i in relevant_indices if i < len(state.articles)]
            logging.info(f"Batch filtering complete. Found {len(state.articles)} relevant articles.")
        except (ValueError, IndexError) as e:
            logging.error(f"Could not parse relevant article indices from model response: {e}")
            # As a fallback, consider all articles relevant if parsing fails
            pass 
            
        return state

    @graph_node("summarize")
    def _summarize_step(self, state: NewsAgentState) -> NewsAgentState:
        """Summarizes all relevant articles in a single batch API call."""
        if not state.articles:
            logging.warning("No articles to summarize.")
            return state

        # Create a single prompt with the full content of all relevant articles
        article_contents = []
        for i, article in enumerate(state.articles):
            article_contents.append(f"Article {i+1}:\nTitle: {article.get('title')}\nContent: {article.get('content')}")

        prompt = (
            "Please provide a concise, 1-2 sentence summary for each of the following articles.\n\n"
            f"{'---'.join(article_contents)}\n\n"
            "Provide the summaries in a numbered list, like this:\n"
            "1. [Summary of Article 1]\n"
            "2. [Summary of Article 2]\n"
        )
        
        response = self.model.generate_content(prompt)
        
        # Parse the response to extract individual summaries
        # This regex looks for lines starting with a number and a dot.
        summaries = re.findall(r'^\d+\.\s*(.*)', response.text, re.MULTILINE)
        
        if not summaries:
            # Fallback if the regex fails: split by newline
            summaries = [line.strip() for line in response.text.split('\n') if line.strip()]

        state.summaries = summaries
        logging.info(f"Batch summarization complete. Generated {len(state.summaries)} summaries.")
        return state
