from langchain_ollama import ChatOllama
from langchain_core.tools import tool
# 💡 FIX 1: Using the new production module location recommended in your terminal logs
from langchain.agents import create_agent
from duckduckgo_search import DDGS

print("🤖 Initializing production-grade agent state controller...")

# 1. Define the web search tool using LangChain's decorator
@tool
def search_the_web(query: str) -> str:
    """Searches the internet for a given query and returns snippets of information."""
    print(f"\n🌍 [Agent Controller calling Web Scraper for: '{query}']")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

# 2. Bind the tool to our local model
tools = [search_the_web]
model = ChatOllama(model="llama3.2:1b", temperature=0)

# 3. Use the standardized system prompt instruction configuration
system_instruction = (
    "You are an expert research assistant. The current date is September 2026. "
    "You must use the `search_the_web` tool to find real-time, current data from 2026 "
    "before writing your response. Always include '2026' or 'latest' in your search queries "
    "to ensure fresh results. Summarize your findings cleanly in a bulleted list."
)

# 💡 FIX 2: Passed the instruction via the standardized 'system_prompt' variable name
app = create_agent(model, tools=tools, system_prompt=system_instruction)

# 4. Execute a structured query
inputs = {"messages": [("user", "What is a recent major headline about Apple right now?")]}

print("\n🚀 Graph Execution Started...")
for chunk in app.stream(inputs, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    if latest_message.type == "ai" and latest_message.content:
        print(f"\n👑 Final Agent Answer:\n{latest_message.content}")
