#  Production-Grade Local AI Research Agent

A lightweight, state-managed autonomous AI agent built using **LangGraph** and **LangChain**. This system runs completely locally using **Ollama** (`Llama 3.2:1b`) and uses a custom tool to scrape real-time internet data from the live web.

---

##  Key Features

* **Local Intelligence:** Runs entirely offline/locally via Ollama, protecting data privacy.
* **Autonomous Tool Calling:** The agent dynamically decides when it needs to search the web using a built-in DuckDuckGo web scraper tool.
* **Stateful Architecture:** Built using the modern `create_agent` framework to handle prompt injections and execution loops cleanly.

---

##  Technology Stack

* **Orchestration:** LangGraph / LangChain
* **Local LLM:** Ollama (`llama3.2:1b`)
* **Search Execution:** DuckDuckGo-Search API (`ddgs`)
* **Language:** Python 3.10+

---

##  Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd ai_agent
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install langchain langgraph langchain-ollama duckduckgo-search
   ```

3. **Install and Run Ollama:**
   Download Ollama from [ollama.com](https://ollama.com) and pull the lightweight model:
   ```bash
   ollama run llama3.2:1b
   ```

4. **Execute the Agent:**
   ```bash
   python app.py
   ```

---

##  Engineering Insights & Model Limitations

During testing with the ultra-lightweight **Llama 3.2:1b** local model, a fascinating edge-case behavior was observed:

* **The Phenomenon:** When asked for major tech headlines, the agent successfully triggered the tool and queried the live web, but generated a synthesized final response mentioning an *"iPhone 21 Pro"* (which does not exist yet).
* **The Engineering Root Cause:** Tiny 1B parameter models have a limited semantic reasoning window. When parsing raw text snippets returned from the live web search scraper, the model struggled to differentiate real factual headlines from speculative forum fluff or clickbait articles.
* **The Scaled Solution:** To patch this in enterprise environments, you would upscale the node to a larger local model (such as `Llama 3:8B`) or implement a **Multi-Agent Architecture** where a secondary Critic/Fact-Checker agent filters the scraper's output against verified sources.
