from langchain_ollama import ChatOllama
from langchain_core.tools import tool
# 💡 FIX 1: Using the updated, warning-free agent function location
from langchain.agents import create_agent
from duckduckgo_search import DDGS

print("🕸️ Initializing multi-agent collaborative network...")

# --- 🛠️ STEP 1: DEFINE THE SCRAPER TOOL ---
@tool
def search_the_web(query: str) -> str:
    """Searches the internet for a given query and returns snippets of information."""
    print(f"\n🌍 [Agent 1 calling Web Scraper for: '{query}']")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

# --- 🤖 STEP 2: INITIALIZE THE BASE MODEL ---
model = ChatOllama(model="llama3.2:1b", temperature=0)

# --- 🎭 STEP 3: CONSTRUCT AGENT 1 (THE RESEARCHER) ---
researcher_prompt = (
    "You are a raw research collection agent. Your only goal is to use the `search_the_web` tool "
    "to find recent context for the user query. Do not write a beautiful essay. "
    "Output the exact raw lines you find."
)
# 💡 FIX 2: Switched Agent 1 to use create_agent and system_prompt
researcher_node = create_agent(model, tools=[search_the_web], system_prompt=researcher_prompt)

# --- 🎭 STEP 4: CONSTRUCT AGENT 2 (THE FACT-CHECKER) ---
fact_checker_prompt = (
    "You are a senior technical supervisor and fact-checker. The current date is September 2026. "
    "Review the raw research text provided by the research agent. "
    "CRITICAL FILTER RULE: Look for temporal chronological contradictions. If the text mentions "
    "products or events far beyond 2026 (like iPhone 21 or future concepts), strip them out "
    "as clickbait or speculative hallucinations. "
    "Deliver a verified, highly accurate factual bulleted summary of true news for 2026."
)
# 💡 FIX 3: Switched Agent 2 to use create_agent and system_prompt
fact_checker_node = create_agent(model, tools=[], system_prompt=fact_checker_prompt)

# --- 🚀 STEP 5: ORCHESTRATE THE AGENT TEAM PIPELINE ---
user_query = "What is a recent major headline about Apple right now?"
inputs = {"messages": [("user", user_query)]}

print("\n🚀 Multi-Agent Node Team Execution Started...")

# Node 1: Run the Researcher to get web contents
print("\n🎬 [Node 1: Dispatching Researcher Agent...]")
research_state = researcher_node.stream(inputs, stream_mode="values")
# Loop through the stream to get the final message content from the generator
for chunk in research_state:
    raw_research_output = chunk["messages"][-1].content

# Node 2: Pass the Researcher's output straight into the Fact-Checker
print("\n🎬 [Node 2: Dispatching Fact-Checker Agent for Validation...]")
fact_checker_inputs = {
    "messages": [
        ("user", f"Verify this raw research data extracted from the web:\n\n{raw_research_output}")
    ]
}
final_state = fact_checker_node.stream(fact_checker_inputs, stream_mode="values")

print("\n👑 Final Verified Team Answer:\n")
for chunk in final_state:
    latest_message = chunk["messages"][-1]
    if latest_message.type == "ai" and latest_message.content:
        print(latest_message.content)
