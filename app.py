import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools import tool
from langchain_tavily import TavilySearch



load_dotenv()


today_date = datetime.now().strftime("%Y-%m-%d")

# Safety check for Tavily API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file")

# Safety check for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# Load predefined simple greetings from a CSV file for quick responses
greetings_responses = {}
try:
    with open('greetings.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            greetings_responses[row['query'].lower()] = row['response']
except FileNotFoundError:
    print("Warning: greetings.csv not found. Custom greetings will not be used.")


embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small"
)


persist_directory = 'chroma_db'
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


retriever = db.as_retriever(search_kwargs={'k': 3})


@tool
def smart_search_tool(query: str) -> str:
    """
    Always use this tool to answer any question about Cristiano Ronaldo.
    It first searches a local database, and if it doesn't find a sufficient answer,
    it automatically searches the internet.
    """

    local_results = retriever.invoke(query)
    

    if local_results:
        context = "\n".join([doc.page_content for doc in local_results])
        return f"Results from local database:\n{context}"

    else:
        tavily_search = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)
        internet_results = tavily_search.invoke(query)
        return f"Results from internet search:\n{internet_results}"


tools = [smart_search_tool]


# --- AGENT AND LLM SETUP ---

# Initialize the Large Language Model (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)

# Initialize chat message history for conversation memory
history = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are 'Al Don Bot', an expert assistant on Cristiano Ronaldo. Always answer in Arabic. Today's date is: {today_date}."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)


# --- MAIN INTERACTION LOOP ---

print("أنا روبوت كريستيانو رونالدو! اسألني أي شيء (اكتب 'خروج' للخروج).")
while True:
    query = input("\nسؤالك: ")
    if query.lower() == "خروج":
        print("وداعاً!")
        break
    
    
    normalized_query = query.strip().lower()
    if normalized_query in greetings_responses:
        print("\nالجواب:", greetings_responses[normalized_query])
        continue
    
    try:
        
        response = agent_executor.invoke({
            "input": query,
            "chat_history": history.messages
        })
        print("\nالجواب:", response["output"])
        
        
        history.add_user_message(query)
        history.add_ai_message(response["output"])
        
    except Exception as e:
        print(f"حدث خطأ أثناء معالجة السؤال: {e}")