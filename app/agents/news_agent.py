import os
import re
import logging
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Constants ---
NEWS_API_URL = "https://eventregistry.org/api/v1/article/getArticles"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Get the key for Google models

# --- Tool Definitions ---

@tool
def search_for_news(query: str) -> str:
    """
    Searches for recent news articles about a specific topic.
    Returns a list of articles including their title, URL, and content as a JSON string.
    The agent should present the titles to the user and ask what to do next.
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

        articles_for_llm = [{
            "title": article.get("title"),
            "content": article.get("body"),
            "url": article.get("url"),
        } for article in raw_articles]

        return json.dumps(articles_for_llm)

    except Exception as e:
        logging.error(f"Error in search_for_news tool: {e}")
        return f"Error searching for news: {e}"


@tool
def summarize_all_and_synthesize(articles_json: str) -> str:
    """
    Receives a JSON string containing a LIST of articles, summarizes each one,
    and then creates a final synthesized overview of all of them.
    Use this when the user asks for a summary of all found articles or a general overview.
    """
    logging.info("Tool 'summarize_all_and_synthesize' called.")
    try:
        articles = json.loads(articles_json)
        if not isinstance(articles, list) or not articles:
            return "Error: Invalid or empty list of articles provided."
    except json.JSONDecodeError:
        return "Error: The input must be a valid JSON string of articles."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)

    article_contents = []
    for i, article in enumerate(articles):
        article_contents.append(f"Article {i+1}:\nTitle: {article.get('title')}\nContent: {article.get('content')}")

    prompt = (
        "You are a world-class news analyst. Your task is to do two things:\n"
        "1. Provide a concise, 1-2 sentence summary for EACH of the following articles.\n"
        "2. After summarizing them all, write a final, 'Key Takeaways' paragraph that synthesizes the main themes from all the articles into a cohesive overview.\n\n"
        "Here are the articles:\n\n"
        f"{'---'.join(article_contents)}\n\n"
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
    Use this when the user wants to dive deep into one specific article from a search list.
    The calling agent is responsible for selecting the correct article from the conversation history
    and passing its JSON object to this tool.
    """
    logging.info("Tool 'summarize_one_article' called.")
    try:
        article = json.loads(article_json)
        if not isinstance(article, dict):
            return "Error: Invalid article format. Input must be a JSON object for a single article."
    except json.JSONDecodeError:
        return "Error: The input must be a valid JSON string of a single article."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)
    title = article.get('title', 'No Title')
    content = article.get('content', 'No Content')

    prompt = (
        f"Please provide a detailed summary of the following news article:\n\n"
        f"Title: {title}\n\n"
        f"Content: {content}\n\n"
        "The summary should capture the key points, main arguments, and any important conclusions."
    )
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        logging.error(f"Error during single article summarization: {e}")
        return f"Error summarizing article: {e}"


# --- Main Conversational Agent ---
def main():
    """
    Sets up and runs the conversational news agent.
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
        return
        
    tools = [
        search_for_news,
        summarize_all_and_synthesize,
        summarize_one_article
    ]
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    
    with SqliteSaver.from_conn_string(":memory:") as memory:
        agent_executor = create_react_agent(model, tools, checkpointer=memory)

        print("Conversational News Agent is ready. Type 'exit' to end the conversation.")
        print("Example: 'What's the latest news on AI regulations?'")

        thread_id = "news-thread-1"
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            try:
                user_input = input("\n> ")
                if user_input.lower() == 'exit':
                    print("Ending conversation. Goodbye!")
                    break

                input_message = {"messages": [("user", user_input)]}

                full_response = ""
                print("\n--- Agent Thinking ---")
                # FIX: This loop now prints the agent's thought process, including tool calls.
                for chunk in agent_executor.stream(input_message, config):
                    # The chunk is a dictionary with a single key, which is the name of the node that was just executed.
                    for key, value in chunk.items():
                        if key == "agent":
                            # This is the agent's turn, where it decides to act or respond.
                            agent_message = value['messages'][-1]
                            if agent_message.tool_calls:
                                # The agent decided to use a tool.
                                print(f"Calling Tool: {agent_message.tool_calls[0]['name']} with args {agent_message.tool_calls[0]['args']}")
                            else:
                                # The agent is responding to the user.
                                if agent_message.content:
                                    print(agent_message.content[len(full_response):], end="", flush=True)
                                    full_response = agent_message.content
                        elif key == "tools":
                            # This is the result of the tool call.
                            # We'll just print a confirmation that the tool ran.
                            print("\nTool Executed. Agent is processing the result...")

                print("\n--- End of Response ---")
            except (KeyboardInterrupt, EOFError):
                print("\nEnding conversation. Goodbye!")
                break


if __name__ == "__main__":
    main()
