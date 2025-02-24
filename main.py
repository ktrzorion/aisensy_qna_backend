from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import BSHTMLLoader
from playwright.async_api import async_playwright
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import logging
from fastapi.middleware.cors import CORSMiddleware
import uuid
from fastapi.staticfiles import StaticFiles
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Web Content Q&A Tool")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing user data and static files if they don't exist
os.makedirs("user_data", exist_ok=True)
os.makedirs("static", exist_ok=True)

# User storage - maps user_id to their data
# In production, this should be a proper database
user_data = {}

# Define request and response models
class ScrapeRequest(BaseModel):
    urls: List[HttpUrl]
    use_playwright: bool = True

class QuestionRequest(BaseModel):
    question: str

class RemoveUrlRequest(BaseModel):
    url: HttpUrl

class Answer(BaseModel):
    answer: str
    source_documents: Optional[List[str]] = None

class UrlsResponse(BaseModel):
    urls: List[str]

# Get or create user_id
async def get_user_id(x_user_id: Optional[str] = Header(None)):
    if not x_user_id:
        x_user_id = str(uuid.uuid4())
    return x_user_id

async def load_urls(urls, use_playwright=True):
    """Load content from URLs using either Playwright or BeautifulSoup (async version)"""
    try:
        documents = []
        
        if use_playwright:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()

                for url in urls:
                    url_str = str(url)  # Convert HttpUrl to string
                    logger.info(f"Fetching: {url_str}")
                    await page.goto(url_str)
                    content = await page.content()
                    
                    # Convert dict to Document
                    documents.append(Document(page_content=content, metadata={"source": url_str}))

                await browser.close()
        else:
            for url in urls:
                loader = BSHTMLLoader(str(url))
                documents.extend(loader.load())

        return documents
    except Exception as e:
        logger.error(f"Error loading URLs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load URLs: {str(e)}")

def process_documents(documents):
    """Split documents into chunks and create embeddings"""
    try:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        
        return vector_store
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

def setup_qa_chain(vector_store):
    """Set up the QA chain with the vector store"""
    try:
        # Initialize OpenAI LLM (ensure OPENAI_API_KEY is set in .env)
        llm = OpenAI(temperature=0)
        
        # Create the retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up QA chain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set up QA chain: {str(e)}")

def save_user_data(user_id: str):
    """Save user vector store to disk"""
    user_dir = f"user_data/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    
    # Save FAISS index and documents metadata
    if user_id in user_data and "vector_store" in user_data[user_id]:
        user_data[user_id]["vector_store"].save_local(user_dir)
        logger.info(f"Saved data for user {user_id}")

def load_user_data(user_id: str):
    """Load user vector store from disk if it exists"""
    user_dir = f"user_data/{user_id}"
    if os.path.exists(user_dir):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vector_store = FAISS.load_local(user_dir, embeddings)
            return vector_store
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
    return None

# API endpoints
@app.post("/scrape", status_code=200)
async def scrape_urls(request: ScrapeRequest, user_id: str = Depends(get_user_id)):
    """Endpoint to scrape and process URLs"""
    try:
        logger.info(f"Scraping URLs for user {user_id}: {request.urls}")
        documents = await load_urls(request.urls, request.use_playwright)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Initialize user data if not exists
        if user_id not in user_data:
            user_data[user_id] = {
                "vector_store": None,
                "qa_chain": None,
                "url_to_docs": {}
            }
        
        # Store documents by URL
        for doc in documents:
            url = doc.metadata["source"]
            if url not in user_data[user_id]["url_to_docs"]:
                user_data[user_id]["url_to_docs"][url] = []
            user_data[user_id]["url_to_docs"][url].append(doc)
        
        # Process all documents
        all_docs = []
        for docs_list in user_data[user_id]["url_to_docs"].values():
            all_docs.extend(docs_list)
        
        # Create vector store and QA chain
        vector_store = process_documents(all_docs)
        qa_chain = setup_qa_chain(vector_store)
        
        # Update user data
        user_data[user_id]["vector_store"] = vector_store
        user_data[user_id]["qa_chain"] = qa_chain
        
        # Save user data to disk
        save_user_data(user_id)
        
        return {
            "message": f"Successfully processed {len(documents)} documents from {len(request.urls)} URLs",
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=Answer)
async def ask_question(request: QuestionRequest, user_id: str = Depends(get_user_id)):
    """Endpoint to ask questions about the scraped content"""
    # Initialize user data if not exists but try to load from disk
    if user_id not in user_data:
        user_data[user_id] = {
            "vector_store": load_user_data(user_id),
            "qa_chain": None,
            "url_to_docs": {}
        }
        
        # Setup QA chain if vector store exists
        if user_data[user_id]["vector_store"]:
            user_data[user_id]["qa_chain"] = setup_qa_chain(user_data[user_id]["vector_store"])
    
    if not user_data[user_id]["qa_chain"]:
        raise HTTPException(status_code=400, detail="No content has been scraped yet. Please call /scrape first.")
    
    try:
        logger.info(f"Processing question for user {user_id}: {request.question}")
        result = user_data[user_id]["qa_chain"]({"query": request.question})
        
        # Extract source documents
        source_docs = []
        if "source_documents" in result:
            source_docs = [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        
        return Answer(
            answer=result["result"],
            source_documents=source_docs
        )
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-url")
async def remove_url(request: RemoveUrlRequest, user_id: str = Depends(get_user_id)):
    """Endpoint to remove a specific URL's data"""
    if user_id not in user_data:
        user_data[user_id] = {
            "vector_store": load_user_data(user_id),
            "qa_chain": None,
            "url_to_docs": {}
        }
    
    if not user_data[user_id]["vector_store"]:
        raise HTTPException(status_code=400, detail="No content has been scraped yet")
    
    url_str = str(request.url)
    
    if url_str not in user_data[user_id]["url_to_docs"]:
        raise HTTPException(status_code=404, detail=f"URL {url_str} not found in user data")
    
    try:
        # Remove the URL from user data
        user_data[user_id]["url_to_docs"].pop(url_str)
        
        # If there are no more URLs, remove all user data
        if not user_data[user_id]["url_to_docs"]:
            user_data[user_id]["vector_store"] = None
            user_data[user_id]["qa_chain"] = None
            
            # Remove user directory
            user_dir = f"user_data/{user_id}"
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                
            return {"message": f"Removed URL {url_str}. All user data has been cleared as no URLs remain."}
        
        # Otherwise, rebuild the vector store with remaining documents
        all_docs = []
        for docs_list in user_data[user_id]["url_to_docs"].values():
            all_docs.extend(docs_list)
        
        # Create vector store and QA chain
        vector_store = process_documents(all_docs)
        qa_chain = setup_qa_chain(vector_store)
        
        # Update user data
        user_data[user_id]["vector_store"] = vector_store
        user_data[user_id]["qa_chain"] = qa_chain
        
        # Save user data to disk
        save_user_data(user_id)
        
        return {"message": f"Successfully removed data for URL {url_str}"}
    except Exception as e:
        logger.error(f"Error removing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/urls", response_model=UrlsResponse)
async def get_urls(user_id: str = Depends(get_user_id)):
    """Endpoint to get all URLs scraped by the user"""
    if user_id not in user_data or not user_data[user_id]["vector_store"]:
        # Try to load from disk
        user_data[user_id] = {
            "vector_store": load_user_data(user_id),
            "qa_chain": None,
            "url_to_docs": {}
        }
        
        if not user_data[user_id]["vector_store"]:
            return UrlsResponse(urls=[])
    
    return UrlsResponse(urls=list(user_data[user_id]["url_to_docs"].keys()))

@app.get("/health")
async def health_check():
    """Endpoint to check if the service is running"""
    return {"status": "healthy"}

# Serve static files (HTML/CSS/JS)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)