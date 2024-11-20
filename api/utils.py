from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import requests
import os
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_pinecone import Pinecone
import asyncio
import nest_asyncio
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
# Load environment variables from .env file
load_dotenv()
# Allow nested asyncio event loops
nest_asyncio.apply()
llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
index_name="boomvectors"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def store_docs_in_pinecone_vs(docs, index_name, embeddings, urls):
#   print(docs, index_name, embeddings)
  send_urls_to_database = add_new_urls_to_database(urls)
  pine_vs = Pinecone.from_documents(documents = docs, embedding = embeddings, index_name=index_name)
#   print(pine_vs)
  return send_urls_to_database

def add_new_urls_to_database(urls):
    """
    Adds new URLs to the database by sending them to an external API endpoint.

    Args:
        urls (list): List of new URLs to be added to the database.

    Returns:
        str: A message indicating the result of the request.
    """
    api_url = f"https://toolbox.boomlive.in/api_project/add_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": "adityaboom_requesting2024#",
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request with the URLs in the payload
        response = requests.get(api_url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            # You can log or process the response data as required
            # noofurls = len(urls)
            # print(urls, noofurls)
            return f"Successfully added URLs to the database."
        else:
            if(len(urls) == 0):
                return f"There are no urls to add"
            return f"There are no urls to add"
    except requests.RequestException as e:
        return f"An error occurred while adding URLs: {e}"


def read_docs(urls):
    docs = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses

            # Check if the content type is HTML
            if 'text/html' not in response.headers.get('Content-Type', ''):
                print(f"Skipped non-HTML content at {url}")
                continue

            # Parse HTML and selectively extract relevant text
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

            # Create Document object with extracted content
            document = Document(
                page_content=text,
                metadata={"source": url}
            )
            docs.append(document)

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            continue
        docs = chunk_data(docs)
    return docs

def chunk_data(docs, chunk_size=1000,chunk_overlap= 200):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(docs)
    return docs

def filter_new_urls(urls):
    """
    Filters the given URLs by sending them to an external API endpoint and 
    receiving the URLs not already in the database.

    Args:
        urls (list): List of URLs to be filtered.

    Returns:
        list: List of new URLs not in the database.
    """
    new_urls = []
    # print("Input URLs:", urls)

    # API endpoint and headers
    # api_url =  f"http://192.168.68.133/boomProject/api_project/not_in_table.php?urls={urls}"

    api_url = f"https://toolbox.boomlive.in/api_project/not_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": "adityaboom_requesting2024#",
        "Content-Type": "application/json"
    }
    try:
        # Send the POST request
        response = requests.get(api_url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            new_urls = response_data.get("urls", [])
            # print("New URLs returned by the API:", new_urls)
        else:
            print(f"Failed to filter URLs. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"An error occurred while filtering URLs: {e}")

    return new_urls



def fetch_article_urls():
    """
    Fetches article URLs from the specified API endpoint.
    """
    # API endpoint and headers
    api_url = "https://boomlive.in/dev/h-api/news"
    headers = {
        "accept": "*/*",
        "s-id": os.getenv("S_ID")  # Ensure S_ID is defined in .env
    }

    try:
        # Send GET request to the API
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            article_urls = []

            # Iterate over the news items and construct the full URL
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path:
                    # full_url = f"https://www.boomlive.in{url_path}"
                    article_urls.append(url_path)
            formatted_urls = json.dumps(article_urls)
            return formatted_urls
        else:
            print(f"Failed to fetch articles. Status code: {response.status_code}")
            return []
    except requests.RequestException as e:
        # Handle exceptions during the request
        print(f"An error occurred while fetching articles: {e}")
        return []

################################Store Old Articles to vector database##############################################################

async def store_old_articles():
    # Initialize variables
    article_urls = []
    start_index = 0
    count = 20

    while True:
        perpageurl = []
        print("Now start index is ",start_index)
        # Construct the API URL with the current startIndex and count
        api_url = f'https://boomlive.in/dev/h-api/news?startIndex={start_index}&count={count}'
        headers = {
            "accept": "*/*",
            "s-id": "1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o"
        }

        # Send GET request to the API
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # If there are no articles, break the loop
            if not data.get("news"):
                break

            # Iterate over the news items and construct the full URL
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path:
                    # full_url = f"https://www.boomlive.in{url_path}"
                    full_url = url_path
                    article_urls.append(url_path)
                    perpageurl.append(url_path)

            filtered_urls = await filter_old_urls(json.dumps(perpageurl))
            # Increment the startIndex by 1 for the next article batch
            docsperindex = await fetch_article_docs(filtered_urls)
            print(f"There are total {len(filtered_urls)} new articles urls and its {len(docsperindex)} new chunked docs to be added in pinecone")

            await store_old_docs_in_pinecone_vs(docsperindex, embeddings, index_name, filtered_urls)
            start_index += 20
            print(f"These are current index {start_index} urls", filtered_urls)
        else:
            print(f"Failed to fetch articles. Status code: {response.status_code}")
            break

    return article_urls


async def fetch_article_docs(urls):
    # Prepare list to store documents
    data = []

    # Initialize text splitter once outside the loop
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses

            # Check if the content type is HTML
            if 'text/html' not in response.headers.get('Content-Type', ''):
                print(f"Skipped non-HTML content at {url}")
                continue

            # Parse HTML and selectively extract relevant text
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

            # Create Document object with extracted content
            document = Document(
                page_content=text,
                metadata={"source": url}
            )
            data.append(document)

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            continue

    # Split documents and return
    # docs = text_splitter.split_documents(data)
    docs = await chunk_old_data(data)
    return docs


async def store_old_docs_in_pinecone_vs(docs, embeddings, index_name, urls):
  pine_vs = Pinecone.from_documents(documents = docs, embedding = embeddings, index_name=index_name)
  print(f"Added {len(docs)} Articles chunks in the pinecone")
  await add_old_urls_to_database(json.dumps(urls))
  return pine_vs


async def filter_old_urls(urls):
    """
    Filters the given URLs by sending them to an external API endpoint and 
    receiving the URLs not already in the database.

    Args:
        urls (list): List of URLs to be filtered.

    Returns:
        list: List of new URLs not in the database.
    """
    new_urls = []
    # print("Input URLs:", urls)

    # API endpoint and headers
    # api_url =  f"http://192.168.68.133/boomProject/api_project/not_in_table.php?urls={urls}"

    api_url = f"https://toolbox.boomlive.in/api_project/not_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": "adityaboom_requesting2024#",
        "Content-Type": "application/json"
    }
    try:
        # Send the POST request
        response = requests.get(api_url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            new_urls = response_data.get("urls", [])
            print("These are new urls to be added in pinecone:", new_urls)
        else:
            print(f"Failed to filter URLs. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"An error occurred while filtering URLs: {e}")

    return new_urls


async def chunk_old_data(docs, chunk_size=1000,chunk_overlap= 200):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(docs)
    return docs


async def add_old_urls_to_database(urls):
    """
    Adds new URLs to the database by sending them to an external API endpoint.

    Args:
        urls (list): List of new URLs to be added to the database.

    Returns:
        str: A message indicating the result of the request.
    """
    api_url = f"https://toolbox.boomlive.in/api_project/add_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": "adityaboom_requesting2024#",
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request with the URLs in the payload
        response = requests.get(api_url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            # You can log or process the response data as required
            # noofurls = len(urls)
            # print(urls, noofurls)
            print(f"Successfully added {len(urls)}URLs to the database." )
            return f"Successfully added URLs to the database."
        else:
            if(len(urls) == 0):
                return f"There are no urls to add"
            return f"There are no urls to add"
    except requests.RequestException as e:
        return f"An error occurred while adding URLs: {e}"

