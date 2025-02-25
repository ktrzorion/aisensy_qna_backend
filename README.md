# Web Content Q&A Tool

A powerful tool that allows users to scrape web content and ask questions about it using AI-powered natural language processing.

## Live Demo

Try it out: [http://13.232.93.150:8001/](http://13.232.93.150:8001/)

## Repositories

- **Backend**: [https://github.com/ktrzorion/aisensy_qna_backend](https://github.com/ktrzorion/aisensy_qna_backend)
- **Frontend**: [https://github.com/ktrzorion/aisensy_qna_frontend](https://github.com/ktrzorion/aisensy_qna_frontend)

## Features

- Scrape content from multiple web URLs simultaneously
- Support for dynamic websites using Playwright
- Ask natural language questions about the scraped content
- AI-powered answers with source references
- Manage scraped URLs (add/remove)
- Persistent storage of user data
- Responsive UI for all devices

## Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **LangChain**: Framework for developing applications powered by language models
- **OpenAI**: For natural language processing and QA capabilities
- **FAISS**: Vector database for efficient similarity search
- **Playwright**: For scraping dynamic web content
- **HuggingFace Embeddings**: For document embedding and semantic search
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server

### Frontend
- **HTML/CSS/JavaScript**: Pure frontend implementation
- **Font Awesome**: For icons
- **Responsive Design**: Works on mobile and desktop

## Installation

### Prerequisites
- Python 3.10+
- Docker (for deployment)
- OpenAI API key

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/ktrzorion/aisensy_qna_backend.git
   cd aisensy_qna_backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Playwright dependencies:
   ```
   playwright install --with-deps
   ```

5. Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

6. Run the application:
   ```
   python main.py
   ```
   
   The server will start on `http://localhost:8000`

### Frontend Setup

1. Clone the repository:
   ```
   git clone https://github.com/ktrzorion/aisensy_qna_frontend.git
   cd aisensy_qna_frontend
   ```

2. For development, open `index.html` in a browser, or serve using a simple HTTP server:
   ```
   # Using Python
   python -m http.server
   
   # Or using Node.js
   npx serve
   ```

## Usage

### Scraping Web Content

1. Navigate to the "Scrape Content" tab
2. Enter one or more URLs to scrape
3. Check/uncheck "Use Playwright" based on whether the content is dynamic
4. Click "Scrape Content"

### Asking Questions

1. Navigate to the "Ask Questions" tab
2. Type your question about the scraped content
3. Click "Ask Question"
4. View the answer and source references

### Managing URLs

1. Navigate to the "Manage URLs" tab
2. View all scraped URLs
3. Remove URLs you no longer need

## API Endpoints

The backend provides the following API endpoints:

- `POST /scrape`: Scrape and process URLs
  ```json
  {
    "urls": ["https://example.com"],
    "use_playwright": true
  }
  ```

- `POST /ask`: Ask a question about scraped content
  ```json
  {
    "question": "What is the main topic of the website?"
  }
  ```

- `DELETE /remove-url`: Remove a specific URL's data
  ```json
  {
    "url": "https://example.com"
  }
  ```

- `GET /urls`: Get all URLs scraped by the user

- `GET /health`: Check if the service is running

## Project Structure

### Backend
```
.
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── user_data/           # Directory for storing user data
```

### Frontend
```
.
├── index.html           # Main HTML file
├── styles.css           # CSS styles
├── script.js            # JavaScript for frontend functionality
```

## How It Works

1. **Scraping Process**:
   - The app uses Playwright (or requests + BeautifulSoup for static sites) to scrape the content
   - HTML content is processed and cleaned
   - Documents are split into chunks for efficient indexing

2. **Embedding and Indexing**:
   - Text chunks are converted to embeddings using HuggingFace's sentence transformers
   - Embeddings are stored in a FAISS vector database for efficient similarity search

3. **Question Answering**:
   - User questions are processed and vectorized
   - Most relevant text chunks are retrieved from the vector database
   - OpenAI's language model generates answers based on the retrieved content
   - Source references are provided for transparency

4. **Data Management**:
   - User data is stored persistently on disk
   - Users can manage their scraped URLs

## Deployment

### Docker Deployment

#### Backend
1. Build the Docker image:
   ```
   docker build -t aisensy_be .
   ```

2. Run the container:
   ```
   docker run -d --network host aisensy_be
   ```

#### Frontend
1. Build the Docker image:
   ```
   docker build -t aisensy_fe .
   ```

2. Run the container:
   ```
   docker run -d -p 8001:80 --name fe aisensy_fe
   ```

### Server Deployment

1. Set up a server with Python 3.10+
2. Clone the repository and install dependencies
3. Install and configure nginx as a reverse proxy
4. Set up a systemd service to run the application
5. Configure firewall to allow required ports

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Playwright](https://playwright.dev/)
- [Font Awesome](https://fontawesome.com/)
