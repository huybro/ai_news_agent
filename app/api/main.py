from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from app.agents.news_agent import NewsAgent
from app.db.models import init_db, Article, Summary
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI News Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/news_agent")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize news agent
news_agent = NewsAgent(os.getenv("GEMINI_API_KEY"))

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class SummaryResponse(BaseModel):
    title: str
    summary: str
    source: str
    relevance_score: Optional[float]

@app.post("/api/query", response_model=List[SummaryResponse])
async def process_query(request: QueryRequest, db: Session = Depends(get_db)):
    """Process a natural language query and return relevant news summaries."""
    try:
        # Process query using the news agent
        result = await news_agent.process_query(request.query)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result["summaries"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/articles")
async def get_articles(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Retrieve recent articles from the database."""
    articles = db.query(Article).offset(skip).limit(limit).all()
    return articles

@app.get("/api/summaries/{article_id}")
async def get_summaries(
    article_id: int,
    db: Session = Depends(get_db)
):
    """Retrieve summaries for a specific article."""
    summaries = db.query(Summary).filter(Summary.article_id == article_id).all()
    return summaries

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 