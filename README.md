# AI News Agent

An intelligent news aggregation and summarization system powered by Gemini API and LangGraph.

## Features

- Real-time news article retrieval and processing
- Chain-of-thought reasoning for intelligent article summarization
- Vectorized keyword filtering for efficient article matching
- Streamlit-based chatbot interface
- End-to-end observability with workflow tracing
- Scalable PostgreSQL backend with Google Cloud integration

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   NEWS_API_KEY=your_news_api_key
   DATABASE_URL=your_postgresql_url
   ```

## Project Structure

```
ai_news_agent/
├── app/
│   ├── api/            # FastAPI endpoints
│   ├── core/           # Core business logic
│   ├── db/             # Database models and migrations
│   ├── agents/         # LLM agent implementations
│   └── ui/             # Streamlit interface
├── tests/              # Test suite
├── .env               # Environment variables
└── requirements.txt   # Project dependencies
```

## Running the Application

1. Start the backend:
   ```bash
   uvicorn app.api.main:app --reload
   ```

2. Launch the Streamlit UI:
   ```bash
   streamlit run app/ui/main.py
   ```

## License

MIT 