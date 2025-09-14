import os
import time
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    # --- Scrape Wikipedia Content ---
    
    # URL for Cristiano Ronaldo's English Wikipedia page
    url = "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Fetch the webpage
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to download the page. Status code: {response.status_code}")
        return

    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content_div = soup.find("div", id='mw-content-text')
    if not main_content_div:
        print("Could not find the main content container of the article.")
        return
        
    page_text_english = main_content_div.getText()

    # --- Split the English Text into Chunks ---

    # Initialize a text splitter to break the article into smaller pieces
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=100, length_function=len)
    english_texts = text_splitter.split_text(page_text_english)
    
    # --- Translate Each Chunk to Arabic ---

    arabic_texts = []
    total_chunks = len(english_texts)
    
    for i, text in enumerate(english_texts):
        try:
            # The print statement below is kept to show progress for this long-running process
            print(f"Translating chunk {i+1}/{total_chunks}...")
            
            # Translate the text using deep-translator
            translated_text = GoogleTranslator(source='auto', target='ar').translate(text)
            arabic_texts.append(translated_text)
            
            # A short delay to be respectful to the translation API
            time.sleep(0.3) 
            
        except Exception as e:
            print(f"Error translating chunk {i+1}: {e}")
            arabic_texts.append("") # Add an empty string on failure

    # --- Create Embeddings and Store in ChromaDB ---

    print("Creating embeddings for the Arabic texts. This may take a moment...")
    
    # Initialize the Hugging Face embedding model (defaults to CPU)
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )

    # Define the directory to save the vector database
    persist_directory = 'chroma_db'
    
    # Create the ChromaDB vector store from the translated texts
    db = Chroma.from_texts(
        texts=arabic_texts,
        embedding=embedding_model,
        collection_name="Ronaldo_Knowledge_Base",
        persist_directory=persist_directory
    )
    
    print(f"Successfully stored {len(arabic_texts)} text chunks in the database: {persist_directory}")



if __name__ == '__main__':
    main()