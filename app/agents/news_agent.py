from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import google.generativeai as genai
from langgraph.graph import StateGraph, END, START
import requests
import os
from datetime import datetime, timedelta
import traceback

class NewsAgentState(BaseModel):
    """State for the news processing agent."""
    query: str
    articles: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    current_step: str = "initial"
    error: Optional[str] = None

class NewsAgent:
    def __init__(self, gemini_api_key: str):
        """Initialize the news agent with Gemini API."""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Create the workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for news processing."""
        workflow = StateGraph(NewsAgentState)
        
        # Define nodes with proper state handling
        def search_wrapper(state: Dict) -> Dict:
            # Convert to NewsAgentState if it's a dict
            state_obj = NewsAgentState(**state) if isinstance(state, dict) else state
            result = self._search_step(state_obj)
            return result.model_dump()
            
        def filter_wrapper(state: Dict) -> Dict:
            # Convert to NewsAgentState if it's a dict
            state_obj = NewsAgentState(**state) if isinstance(state, dict) else state
            result = self._filter_step(state_obj)
            return result.model_dump()
            
        def summarize_wrapper(state: Dict) -> Dict:
            # Convert to NewsAgentState if it's a dict
            state_obj = NewsAgentState(**state) if isinstance(state, dict) else state
            result = self._summarize_step(state_obj)
            return result.model_dump()
        
        # Add nodes with wrappers
        workflow.add_node("search", search_wrapper)
        workflow.add_node("filter", filter_wrapper)
        workflow.add_node("summarize", summarize_wrapper)
        
        # Define edges
        workflow.add_edge(START, "search")
        workflow.add_edge("search", "filter")
        workflow.add_edge("filter", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return relevant news summaries."""
        print("\n[DEBUG] Starting process_query")
        print(f"[DEBUG] Query: {query}")
        
        # Initialize state as NewsAgentState instance
        initial_state = NewsAgentState(
            query=query,
            articles=[],
            summaries=[],
            current_step="initial",
            error=None
        )
        
        try:
            # Execute the workflow with dictionary state
            final_state = None
            for current_state in self.workflow.stream(initial_state.model_dump()):
                # Convert dict back to NewsAgentState for proper attribute access
                state_obj = NewsAgentState(**current_state)
                print(f"\n[DEBUG] Current step: {state_obj.current_step}")
                print(f"[DEBUG] Articles count: {len(state_obj.articles)}")
                print(f"[DEBUG] Summaries count: {len(state_obj.summaries)}")
                if state_obj.error:
                    print(f"[DEBUG] Error: {state_obj.error}")
                
                final_state = current_state
            
            if final_state is None:
                print("[DEBUG] Workflow did not complete")
                return {"error": "Workflow did not complete", "summaries": []}
            
            # Convert final state back to NewsAgentState
            final_state_obj = NewsAgentState(**final_state)
            return final_state_obj.model_dump()
            
        except Exception as e:
            print(f"[DEBUG] Error in workflow execution: {str(e)}")
            traceback.print_exc()
            return {
                "error": f"Workflow execution error: {str(e)}",
                "summaries": [],
                "articles": [],
                "current_step": "execution_error",
                "query": query
            } 

    def _search_step(self, state: NewsAgentState) -> NewsAgentState:
        """Handle search step in the workflow."""
        print("\n[DEBUG] Starting search step")
        try:
            if not self.news_api_key:
                raise ValueError("NEWS_API_KEY environment variable is not set")

            # Use state.query directly
            payload = {
                "action": "getArticles",
                "keyword": state.query,  # Direct attribute access
                "sourceLocationUri": [
                    "http://en.wikipedia.org/wiki/United_States",
                    "http://en.wikipedia.org/wiki/Canada",
                    "http://en.wikipedia.org/wiki/United_Kingdom"
                ],
                "ignoreSourceGroupUri": "paywall/paywalled_sources",
                "articlesPage": 1,
                "articlesCount": 5,
                "articlesSortBy": "date",
                "articlesSortByAsc": False,
                "dataType": ["news", "pr"],
                "forceMaxDataTimeWindow": 31,
                "resultType": "articles",
                "apiKey": self.news_api_key,
                "articleBodyLen": -1,
                "includeArticleBody": True,
                "includeArticleTitle": True,
                "includeArticleBasicInfo": True,
                "includeArticleImage": True,
                "includeArticleAuthors": True,
                "includeArticleSentiment": True
            }

            print("[DEBUG] Making API request to EventRegistry...")
            response = requests.post(
                "https://eventregistry.org/api/v1/article/getArticles",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"NewsAPI request failed with status {response.status_code}: {response.text}")

            data = response.json()
            print(f"[DEBUG] API Response status: {response.status_code}")
            
            if "articles" not in data:
                raise Exception("No articles found in the response")

            # Process and store the articles
            articles = []
            for article in data["articles"]["results"]:
                processed_article = {
                    "title": article.get("title", ""),
                    "content": article.get("body", ""),
                    "source": article.get("source", {}).get("title", "Unknown"),
                    "url": article.get("url", ""),
                    "published_at": article.get("dateTime", ""),
                    "sentiment": article.get("sentiment", 0),
                    "authors": article.get("authors", []),
                    "image": article.get("image", "")
                }
                articles.append(processed_article)

            print(f"[DEBUG] Found {len(articles)} articles")
            
            # Update state directly
            state.articles = articles
            state.current_step = "search_complete"
            state.error = None
            
            return state
            
        except Exception as e:
            print(f"[DEBUG] Search step error: {str(e)}")
            traceback.print_exc()
            state.error = f"Search error: {str(e)}"
            state.current_step = "search_error"
            return state

    def _filter_step(self, state: NewsAgentState) -> NewsAgentState:
        """Handle filter step in the workflow."""
        print("\n[DEBUG] Starting filter step")
        try:
            # Check error directly
            if state.error:
                print(f"[DEBUG] Skipping filter due to previous error: {state.error}")
                return state
            
            if not state.articles:
                raise ValueError("No articles to filter")

            filtered_articles = []
            for i, article in enumerate(state.articles):
                print(f"[DEBUG] Filtering article {i+1}/{len(state.articles)}: {article['title']}")
                
                # Create a preview of the article to avoid large content
                title = article['title']
                content_preview = article['content'][:300] + "..." if len(article['content']) > 300 else article['content']
                
                prompt = f"""Rate the relevance of this article to the query: "{state.query}"
                
                Article Title: {title}
                Article Preview: {content_preview}
                
                Rate from 0-10 where 0 is completely irrelevant and 10 is highly relevant.
                Return only a number from 0-10.
                """
                
                try:
                    response = self.model.generate_content(prompt)
                    response_text = response.text.strip() if response.text else ""
                    
                    # Try to extract number from response
                    try:
                        import re
                        score_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
                        score = float(score_match.group(1)) if score_match else 0
                    except:
                        # Default: keep article if "relevant" is in response
                        score = 5 if "relevant" in response_text.lower() else 0
                    
                    # Keep articles with score of 5 or higher (on scale of 0-10)
                    if score >= 5:
                        article["relevance_score"] = score
                        filtered_articles.append(article)
                        print(f"[DEBUG] Article {i+1} considered relevant with score {score}")
                    else:
                        print(f"[DEBUG] Article {i+1} considered irrelevant with score {score}")
                except Exception as e:
                    print(f"[DEBUG] Error processing article {i+1}: {str(e)}")
                    # If we can't evaluate relevance, keep the article by default
                    article["relevance_score"] = 5
                    filtered_articles.append(article)
            
            print(f"[DEBUG] Filtered to {len(filtered_articles)} relevant articles")
            
            # Update state directly
            state.articles = filtered_articles
            state.current_step = "filtering_complete"
            state.error = None
            
            return state
            
        except Exception as e:
            state.error = f"Filter error: {str(e)}"
            state.current_step = "filter_error"
            return state

    def _summarize_step(self, state: NewsAgentState) -> NewsAgentState:
        """Handle summarize step in the workflow."""
        print("\n[DEBUG] Starting summarize step")
        try:
            # Check if there's already an error
            if state.error:
                print(f"[DEBUG] Skipping summarize due to previous error: {state.error}")
                return state
            
            if not state.articles:
                raise ValueError("No articles to summarize")

            summaries = []
            for article in state.articles:
                print(f"[DEBUG] Summarizing article: {article['title']}")
                
                prompt = f"""Summarize the article: "{article['title']}"
                
                Article Content: {article['content']}
                """
                
                try:
                    response = self.model.generate_content(prompt)
                    response_text = response.text.strip() if response.text else ""
                    
                    summaries.append(response_text)
                    print(f"[DEBUG] Summarized article: {response_text}")
                except Exception as e:
                    print(f"[DEBUG] Error summarizing article: {str(e)}")
                    # If we can't summarize, keep the original article content
                    summaries.append(article['content'])
            
            print(f"[DEBUG] Summarized {len(summaries)} articles")
            
            # Update state directly
            state.summaries = summaries
            state.current_step = "summarizing_complete"
            state.error = None
            
            return state
            
        except Exception as e:
            print(f"[DEBUG] Summarize step error: {str(e)}")
            traceback.print_exc()
            state.error = f"Summarize error: {str(e)}"
            state.current_step = "summarize_error"
            return state 