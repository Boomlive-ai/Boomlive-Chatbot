from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv
from articleOperations import StoreDailyArticles  # Import StoreDailyArticles logic

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize the FastAPI app
app = FastAPI(
    title="Boomlive Chatbot Server",
    version="1.0",
    description="A Chatbot to resolve any user queries"
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


@app.get("/")
async def check_server():
    return "The server is running"


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)
