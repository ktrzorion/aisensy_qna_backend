from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import asyncio
import os
import subprocess
from bs4 import BeautifulSoup
import aiohttp
from dotenv import load_dotenv
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

# Progress tracking - maps user_id to progress data
progress_data = {}

# WebSocket connections - maps user_id to active websocket connections
active_connections = {}

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

class ProgressUpdate(BaseModel):
    current: int
    total: int
    status: str
    url: Optional[str] = None

# Get or create user_id
async def get_user_id(x_user_id: Optional[str] = Header(None)):
    if not x_user_id:
        x_user_id = str(uuid.uuid4())
    return x_user_id

# WebSocket endpoint for progress updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    
    # Store the connection
    active_connections[user_id] = websocket
    
    try:
        # Send initial progress if exists
        if user_id in progress_data:
            await websocket.send_json(progress_data[user_id])
        
        # Keep connection open to receive messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if user_id in active_connections:
            del active_connections[user_id]

# Function to update progress and send to client
async def update_progress(user_id: str, current: int, total: int, status: str, url: Optional[str] = None):
    progress = {
        "current": current,
        "total": total,
        "status": status,
        "url": url
    }
    
    # Store progress data
    progress_data[user_id] = progress
    
    # Send to client if websocket is connected
    if user_id in active_connections:
        try:
            await active_connections[user_id].send_json(progress)
        except Exception as e:
            logger.error(f"Error sending progress update: {e}")

async def load_urls(urls, use_playwright=True, user_id=None):
    try:
        documents = []
        total_urls = len(urls)
        
        if use_playwright:
            # Launch browser with optimized settings
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,  # Ensure headless mode
                    args=['--disable-gpu', '--disable-dev-shm-usage', '--disable-extensions']  # Performance args
                )
                
                # Use a single browser context for all pages
                context = await browser.new_context(
                    java_script_enabled=True,
                    bypass_csp=True,  # Bypass Content Security Policy
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                
                # Process URLs concurrently with limit
                tasks = []
                for i, url in enumerate(urls):
                    url_str = str(url)
                    tasks.append(process_url_with_playwright(context, url_str, i, total_urls, user_id))
                
                # Process up to 3 URLs concurrently
                chunk_size = 3
                for i in range(0, len(tasks), chunk_size):
                    chunk_results = await asyncio.gather(*tasks[i:i+chunk_size])
                    documents.extend([doc for doc in chunk_results if doc])
                
                await context.close()
                await browser.close()
        else:
            # Process URLs concurrently with requests
            tasks = []
            for i, url in enumerate(urls):
                url_str = str(url)
                tasks.append(process_url_with_requests(url_str, i, total_urls, user_id))
            
            # Process up to 5 URLs concurrently
            chunk_size = 5
            for i in range(0, len(tasks), chunk_size):
                chunk_results = await asyncio.gather(*tasks[i:i+chunk_size])
                documents.extend([doc for doc in chunk_results if doc])
                    
        return documents
        
    except Exception as e:
        logger.error(f"Error loading URLs: {e}")
        if user_id:
            await update_progress(user_id, 0, total_urls, "error", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load URLs: {str(e)}")
    
async def process_url_with_playwright(context, url, index, total, user_id=None):
    try:
        logger.info(f"Fetching with Playwright: {url}")
        
        if user_id:
            await update_progress(user_id, index, total, "fetching", url)
        
        page = await context.new_page()
        
        # Set shorter timeouts
        page.set_default_timeout(15000)  # 15 seconds for all operations
        
        # Navigate with timeout
        response = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        
        if not response or response.status >= 400:
            logger.warning(f"Failed to load {url}: {response.status if response else 'No response'}")
            await page.close()
            return None
            
        # Wait for main content to load, but with shorter timeout
        try:
            # Wait for the body to be available
            await page.wait_for_selector("body", timeout=5000)
            
            # Execute script to get page content
            content = await page.content()
            
            # Extract main text content as a fallback
            text_content = await page.evaluate("""() => {
                // Try to find main content
                const article = document.querySelector('article') || 
                               document.querySelector('main') || 
                               document.querySelector('.content') ||
                               document.querySelector('#content');
                               
                return article ? article.innerText : document.body.innerText;
            }""")
            
            # Close the page
            await page.close()
            
            if user_id:
                await update_progress(user_id, index+1, total, "complete", url)
            
            # Return combined document
            return Document(
                page_content=text_content, 
                metadata={"source": url, "html": content[:10000]}  # Store some HTML for context
            )
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            await page.close()
            return None
            
    except Exception as e:
        logger.error(f"Error processing {url} with Playwright: {e}")
        if user_id:
            await update_progress(user_id, index, total, "error", f"{url}: {str(e)}")
        return None
    
# Helper function for processing a URL with requests
async def process_url_with_requests(url, index, total, user_id=None):
    try:
        logger.info(f"Fetching with requests: {url}")
        
        if user_id:
            await update_progress(user_id, index, total, "fetching", url)
        
        # Use aiohttp instead of requests for async
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15) as response:
                if response.status >= 400:
                    logger.warning(f"Failed to load {url}: {response.status}")
                    return None
                    
                html = await response.text()
                
                # Process with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try to find main content
                main_content = soup.find('article') or soup.find('main') or soup.find('.content') or soup.find('#content')
                
                # Extract text content
                text = (main_content or soup).get_text(separator='\n', strip=True)
                
                if user_id:
                    await update_progress(user_id, index+1, total, "complete", url)
                
                return Document(page_content=text, metadata={"source": url})
                
    except Exception as e:
        logger.error(f"Error processing {url} with requests: {e}")
        if user_id:
            await update_progress(user_id, index, total, "error", f"{url}: {str(e)}")
        return None

async def process_documents(documents, user_id=None):
    """Split documents into chunks and create embeddings"""
    try:
        # Update progress
        if user_id:
            await update_progress(user_id, 0, 100, "processing", "Splitting documents...")
            
        # Optimize text splitting with more selective content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,  # Reduced overlap
            separators=["\n\n", "\n", ". ", " ", ""]  # Explicit separators
        )
        
        # Process in smaller batches to avoid memory issues
        all_splits = []
        batch_size = 10
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            splits = text_splitter.split_documents(batch)
            all_splits.extend(splits)
            
            # Update progress periodically
            if user_id:
                progress = min(40, int(40 * (i + batch_size) / len(documents)))
                await update_progress(user_id, progress, 100, "processing", f"Splitting documents: {i+batch_size}/{len(documents)}")
        
        # Update progress
        if user_id:
            await update_progress(user_id, 50, 100, "processing", "Creating embeddings...")
        
        # Use a smaller, faster embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, faster model
            cache_folder="embeddings_cache"  # Cache embeddings
        )
        
        # Create vector store in batches
        vector_store = None
        batch_size = 100  # Process 100 splits at a time
        
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i+batch_size]
            
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                batch_vector = FAISS.from_documents(batch, embeddings)
                vector_store.merge_from(batch_vector)
            
            # Update progress periodically
            if user_id:
                progress = 50 + min(50, int(50 * (i + batch_size) / len(all_splits)))
                await update_progress(user_id, progress, 100, "processing", f"Creating embeddings: {i+batch_size}/{len(all_splits)}")
        
        # Update progress
        if user_id:
            await update_progress(user_id, 100, 100, "complete", "Processing complete!")
            
        return vector_store
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        
        # Update progress with error
        if user_id:
            await update_progress(user_id, 0, 100, "error", str(e))
            
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

def setup_qa_chain(vector_store):
    """Set up the QA chain with the vector store"""
    try:
        # Initialize OpenAI LLM (ensure OPENAI_API_KEY is set in .env)
        # llm = OpenAI(temperature=0)
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
        
        # Reset progress
        await update_progress(
            user_id=user_id,
            current=0,
            total=len(request.urls),
            status="starting",
            url="Initializing scraper..."
        )
        
        # Load URLs with progress updates
        documents = await load_urls(request.urls, request.use_playwright, user_id)
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
        
        # Create vector store and QA chain with progress updates
        vector_store = await process_documents(all_docs, user_id)
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
        
        # Update progress with error
        await update_progress(
            user_id=user_id,
            current=0,
            total=len(request.urls),
            status="error",
            url=str(e)
        )
        
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
        
        # Update progress
        await update_progress(
            user_id=user_id,
            current=0,
            total=100,
            status="processing",
            url="Rebuilding index after URL removal..."
        )
        
        # Create vector store and QA chain
        vector_store = await process_documents(all_docs, user_id)
        qa_chain = setup_qa_chain(vector_store)
        
        # Update user data
        user_data[user_id]["vector_store"] = vector_store
        user_data[user_id]["qa_chain"] = qa_chain
        
        # Save user data to disk
        save_user_data(user_id)
        
        return {"message": f"Successfully removed data for URL {url_str}"}
    except Exception as e:
        logger.error(f"Error removing URL: {e}")
        
        # Update progress with error
        await update_progress(
            user_id=user_id,
            current=0,
            total=100,
            status="error",
            url=str(e)
        )
        
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

# Ensure Playwright is installed
subprocess.run(["playwright", "install", "--with-deps"], check=True)

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)