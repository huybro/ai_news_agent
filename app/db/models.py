from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(200))
    url = Column(String(500), unique=True)
    published_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Vector embeddings for semantic search
    embedding = Column(Text)  # Store as JSON string
    
    # Relationships
    summaries = relationship("Summary", back_populates="article")
    
    def __repr__(self):
        return f"<Article(title='{self.title}', source='{self.source}')>"

class Summary(Base):
    __tablename__ = "summaries"
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    summary_text = Column(Text, nullable=False)
    relevance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article = relationship("Article", back_populates="summaries")
    
    def __repr__(self):
        return f"<Summary(article_id={self.article_id}, relevance_score={self.relevance_score})>"

# Create tables
def init_db(database_url: str):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine 