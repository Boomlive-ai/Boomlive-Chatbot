# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOpenAI

# from langchain_pinecone import Pinecone
# import os
# from utils import read_docs, chunk_data, filter_new_urls, fetch_article_urls, store_docs_in_pinecone_vs, add_new_urls_to_database
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings

# llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
# index_name="boomvectors"
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# system_prompt = (
#     "Prvoide the summary of article which you found user question similar to article. "
#     "If you don't know the answer, say you don't know. "
#     "Context: {context}"
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )


# def store_docs_in_pinecone_vs(docs, index_name, embeddings):
#   pine_vs = Pinecone.from_documents(documents = docs, embedding = embeddings, index_name=index_name)
#   return pine_vs

# def get_answer_and_source(input_query):
#     pine_index = store_docs_in_pinecone_vs(docs)
#     # Retrieve context based on query embeddings
#     retriever = pine_index.as_retriever()
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     chain = create_retrieval_chain(retriever, question_answer_chain)
#     context_docs = retrieve_context_from_query(input_query, pine_index)

#     # Format context for the chain prompt with improved readability
#     if context_docs:
#         context = "\n\n".join([f"Source: {doc['metadata']['source']}\n{doc['content']}" for doc in context_docs])
#     else:
#         context = "No relevant context found."

#     # Generate the response
#     response = chain.invoke({"input": input_query, "context": context})

#     # Extract the answer
#     answer = response.get("answer", "I'm sorry, I don't have an answer to your question.")

#     # Check if the answer is a fallback response that shouldn't have sources
#     fallback_phrases = [
#         "I cannot fact-check", "I don't know", "I'm sorry", "I don’t have information on", "I'm glad"
#     ]
#     if any(phrase in answer for phrase in fallback_phrases):
#         # If it's a fallback answer, do not include sources
#         source_links = None
#     else:
#         # Gather sources from context_docs first
#         relevant_sources = [doc['metadata']['source'] for doc in context_docs]

#         # Then add unique sources from response context
#         response_context = response.get("context", [])
#         response_sources = [
#             doc.metadata['source'] for doc in response_context if doc.metadata['source'] not in relevant_sources
#         ]

#         # Combine and deduplicate sources, with context_docs sources at the start
#         source_links = relevant_sources + response_sources if relevant_sources else None

#     return {"answer": answer, "source_links": source_links}

# def retrieve_context_from_query(query, vector_store, top_k=1):
#     # Convert user query into an embedding
   
#     similar_docs = pine_index.similarity_search(query, k=top_k)

#     # Extract relevant content and metadata from the most similar document(s)
#     context = [{"content": doc.page_content, "metadata": doc.metadata} for doc in similar_docs]
#     return context

from langchain_pinecone import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Set up LLM and Embeddings
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone Index Configuration (hardcoded index name for reuse)
# INDEX_NAME = "boomvectors"
pine_index = Pinecone(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

# Prompt for the chain
system_prompt = (
    "Provide a summary of the article related to the user's question. "
    "If you don't know the answer, respond with 'I don't know.' "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Function to answer user queries
def get_answer_and_source(input_query, top_k=3):
    """
    Answers user queries using the Pinecone index and LLM for context-based answers.

    Args:
        input_query (str): The user's query.
        top_k (int): Number of top results to retrieve from the index.

    Returns:
        dict: Contains the answer and source links, if any.
    """
    try:
        # Create a retriever from the Pinecone index
        retriever = pine_index.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        
        # Fetch context documents relevant to the query
        similar_docs = retriever.get_relevant_documents(input_query)
        context = "\n\n".join([f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in similar_docs])

        # Create the chain for answering the question
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        # Generate a response
        response = chain.invoke({"input": input_query, "context": context})
        answer = response.get("answer", "I'm sorry, I don't have an answer to your question.")

        # Gather sources from retrieved documents
        source_links = [doc.metadata['source'] for doc in similar_docs] if similar_docs else None

        return {"answer": answer, "source_links": source_links}

    except Exception as e:
        return {"error": str(e)}
