# 🤖 CR7-Chatbot: The Ultimate AI Agent for Cristiano Ronaldo Fans

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-182333?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B7?style=for-the-badge&logo=google-gemini&logoColor=white)

An intelligent, Retrieval-Augmented Generation (RAG) based AI agent built with Python and LangChain to answer any question about Cristiano Ronaldo's illustrious career. This bot leverages a local knowledge base and real-time web search to provide accurate, up-to-date, and grounded answers, ensuring it avoids making things up.

## ✨ Key Features

- **Intelligent Agent:** Powered by **Google's Gemini 1.5 Flash** model, the agent can reason, make decisions, and interact with tools reliably.
- **RAG Architecture:** Utilizes a Retrieval-Augmented Generation pipeline to ensure answers are based on factual data, minimizing hallucinations.
- **Hybrid Search Strategy:** Features a custom `smart_search_tool` that:
    1. First queries a local **ChromaDB** vector database (built from a Wikipedia scrape).
    2. If no sufficient answer is found, it automatically falls back to a **Tavily web search** for real-time information.
- **Conversational Memory:** Remembers the context of the conversation to answer follow-up questions accurately.
- **Robust & Efficient:** Built with professional practices like environment variable management, error handling, and a clean, modular structure.

## ⚙️ How It Works

This project is a practical implementation of a modern AI agent using a Tool Calling architecture.

1.  **Data Ingestion (`ingest.py`):**
    - Scrapes Cristiano Ronaldo's English Wikipedia page using `BeautifulSoup`.
    - Translates the content to Arabic using the reliable `deep-translator` library.
    - Splits the translated text into manageable chunks.
    - Creates vector embeddings for each chunk using `HuggingFaceEmbeddings` (`multilingual-e5-small`).
    - Stores these embeddings in a persistent `ChromaDB` vector database.

2.  **Interaction (`app.py`):**
    - A user asks a question in Arabic.
    - The **Tool Calling Agent**, powered by Gemini, analyzes the query.
    - The agent decides whether to answer from its own internal knowledge (for simple facts) or to call the `smart_search_tool`.
    - The `smart_search_tool` searches the local ChromaDB. If the results are insufficient, it triggers a Tavily web search.
    - The retrieved information (context) is passed back to the Gemini model.
    - The model generates a final, accurate answer in Arabic, grounded in the provided context.

## 🛠️ Tech Stack

- **Core Framework:** Python 3.11+
- **AI/LLM Framework:** LangChain
- **LLM:** Google Gemini 1.5 Flash
- **Vector Database:** ChromaDB
- **Embedding Model:** Hugging Face `intfloat/multilingual-e5-small`
- **Real-time Search:** Tavily Search API
- **Data Processing:** Deep-Translator, BeautifulSoup, Requests
- **Configuration:** python-dotenv

## 🚀 Getting Started

Follow these steps to run the chatbot on your local machine.

### 1. Prerequisites

- Python 3.11 or higher
- Git

### 2. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mithgal-JH/CR7-Chatbot.git](https://github.com/Mithgal-JH/CR7-Chatbot.git)
    cd CR7-Chatbot
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your API Keys:**
    - Create a new file named `.env` in the root of the project.
    - Add your API keys to this file:
      ```env
      GOOGLE_API_KEY="AIzaSy..."
      TAVILY_API_KEY="tvly-..."
      ```

### 3. Build the Knowledge Base

Before running the chatbot for the first time, you need to ingest the data. Run the ingestion script once:
```bash
python ingest.py
```
This will create the `chroma_db` folder containing the vector database.

### 4. Run the Chatbot

You are now ready to chat with the Don Bot!
```bash
python app.py
```

## ⚔️ The Development Journey: A Real-World Case Study

This project was a journey through the real-world challenges of building modern AI applications. The "grind" was real, and several "battles" were fought and won to achieve the final stable version:

- **The Unstable API Battle:** The project initially struggled with unreliable APIs. `googletrans` proved unstable for large texts, and various LLM endpoints (GitHub's experimental API, Groq's rapidly changing models) caused constant `RateLimit` and `Decommissioned` errors. **Victory:** Pivoting to stable, professional services like `deep-translator` and the official **Google Gemini API** provided the necessary reliability to build a functional application.
- **The Dependency Hell Battle:** Rebuilding the environment revealed numerous version conflicts between major libraries (`openai`, `httpx`, etc.). **Victory:** A clean, minimal `requirements.txt` was curated, and the environment was rebuilt from scratch to resolve all conflicts, ensuring reproducibility.
- **The Agent's "Civil War" Battle:** The initial `ReAct` agent constantly failed to follow the strict output format required for tool use, leading to parsing errors and infinite loops. **Victory:** The architecture was fundamentally changed to a **Tool Calling Agent**, which relies on the model's native function-calling capabilities instead of fragile text parsing. This single change eliminated almost all agent-related errors and dramatically improved stability and performance.

This repository stands as a practical example of iterative development and robust AI agent design.
