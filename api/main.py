from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv
from articleOperations import StoreDailyArticles, StoreOldArticles  # Import StoreDailyArticles logic
from chatbotApi import get_answer_and_source
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize the FastAPI app
app = FastAPI(
    title="Boomlive Chatbot Server",
    version="1.0",
    description="A Chatbot to resolve any user queries"
)

application = app

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://yourdomain.com",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the GET request for /storeDailyArticles
@app.get("/storeDailyArticles")
async def store_daily_articles():
    """
    Endpoint to fetch and return daily article URLs.
    """
    # Create an instance of the StoreDailyArticles Runnable
    store_articles = StoreDailyArticles()
    
    # Invoke the logic to fetch articles
    result = store_articles.invoke()

    return result  # Return the result as JSON


@app.get("/query")
async def query_bot(question: str):
    """
    Endpoint to answer user queries.
    """
    result = get_answer_and_source(question)
    return result

@app.post("/store_old_articles")
async def store_old_articles():
    """
    Endpoint to store old articles data from boomlive server to vector database(pinecone)
    """

    store_old_articles = StoreOldArticles()

    result = await store_old_articles.invoke()
    return result



@app.get("/")
async def check_server():
    return "The server is running"


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)