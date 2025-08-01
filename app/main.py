from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

# Import the new agent creation function and database modules
from app.news_agent import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.load import dumps, loads


# Remove the custom serializer - use the default one
# The default serializer handles all LangChain types properly


# --- App Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the creation and teardown of resources for the app.
    This is the correct place to handle database connections.
    """
    # Get database URIs from environment variables
    POSTGRES_URI = os.getenv("POSTGRES_URI")
    
    # Use 'async with' for asynchronous context managers
    async with AsyncPostgresStore.from_conn_string(POSTGRES_URI) as store, \
         AsyncPostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
        
        # In a real production app, you might run setup() once on startup.
        # This creates the necessary tables in your database.
        # If you're getting serialization errors, you may need to drop existing tables first:
       # await checkpointer.drop_tables()  # Uncomment this line to clear existing data
        #await store.setup()
        #await checkpointer.setup()
        
        # Create the agent executor and attach it to the app's state
        agent_executor = create_agent(checkpointer, store)
        app.state.agent_executor = agent_executor
        
        # 'yield' signifies the app is running
        yield
    
    # Code after 'yield' runs on app shutdown (connections are closed by 'with')

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# Get the path to the static directory relative to this file for robustness
static_dir = Path(__file__).parent / "static"

# Mount the 'static' directory to serve the index.html file
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve the main HTML page."""
    with open(static_dir / "index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# This is now an asynchronous generator
async def stream_agent_response(agent_executor, message: str, thread_id: str):
    """Generator function to stream the agent's response."""
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("user", message)]}
    
    full_response = ""
    # Use the asynchronous 'astream' method
    async for chunk in agent_executor.astream(input_message, config):
        print(f"Chunk received: {type(chunk)}")
        
        if agent_response := chunk.get("agent"):
            print(f"Agent response: {type(agent_response)}")
            
            if messages := agent_response.get("messages"):
                print(f"Messages: {len(messages)} messages found")
                
                # Get the last message
                last_message = messages[-1]
                
                # Handle different message types properly
                try:
                    if isinstance(last_message, (AIMessage, HumanMessage, ToolMessage)):
                        content = last_message.content
                    elif hasattr(last_message, 'content'):
                        content = last_message.content
                    else:
                        content = str(last_message)
                    
                    # Ensure content is a string
                    if content is None:
                        content = ""
                    elif not isinstance(content, str):
                        content = str(content)
                        
                except Exception as msg_error:
                    print(f"Error extracting message content: {msg_error}")
                    content = ""
                
                # Only send new content to avoid duplication
                if content and isinstance(content, str):
                    new_content = content[len(full_response):]
                    if new_content:
                        yield new_content
                        full_response = content
                        
        await asyncio.sleep(0.01)

# This is now an asynchronous endpoint
@app.post("/chat")
async def chat_endpoint(request: Request):
    """Endpoint to handle chat requests and stream responses."""
    try:
        data = await request.json()
        chat_request = ChatRequest(**data)
        
        # Get the agent executor from the app's state
        agent_executor = request.app.state.agent_executor
        
        return StreamingResponse(
            stream_agent_response(agent_executor, chat_request.message, chat_request.thread_id),
            media_type="text/plain"
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        # Return error as plain text stream
        async def error_generator():
            yield f"Error: {str(e)}"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/plain"
        )