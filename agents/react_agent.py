"""
ReAct Agent — clean compatibility build with Google Search + live weather
- Works on Python 3.10–3.13
- Supports modern LangChain (≥1.1)
- Tools included:  optional Google (SerpAPI), optional DuckDuckGo, optional Wikipedia, and `weather_now` (Open‑Meteo, no API key)
- Backends supported: OpenAI (ChatGPT), Gemini (Google AI Studio), DeepSeek, LLama
"""

import argparse
import os
import time
from datetime import datetime, timezone
from typing import Any, Union, List, Optional
import unicodedata 
import re
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from dataclasses import dataclass
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage

import scapy.all as scapy
from scapy.sendrecv import AsyncSniffer
PCAP_DIR = "./pcap/react/"

# -----------------------------
# Agent APIs 
# -----------------------------

# For debugging and statistic purposes, we count the number of calls to each tool
tool_call_counts = {}

def _clean_city(s: str) -> str:
    # Normalize unicode, strip smart quotes/whitespace, collapse spaces
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r'^[\s\'"]+|[\s\'"]+$', '', s)  # trim leading/trailing quotes & spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Common alias fix
    s = s.replace(", UK", ", United Kingdom")
    s = s.replace(", U.K.", ", United Kingdom")
    return s

@tool
def weather_now(city: str) -> str:
    """Fetch current weather for a city using Open-Meteo (no API key)."""

    tool_call_counts["weather_now"] = tool_call_counts.get("weather_now", 0) + 1

    try:
        q = _clean_city(city)
        # Fallback: if geocoder fails, try the part before the first comma
        candidates = [q]
        if "," in q:
            candidates.append(_clean_city(q.split(",", 1)[0]))

        geo = None
        for name in candidates:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": name, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            data = resp.json()
            if data.get("results"):
                geo = data
                break

        if not geo or not geo.get("results"):
            return f"No coordinates found for '{q}'."

        g0 = geo["results"][0]
        lat, lon = g0["latitude"], g0["longitude"]
        place = g0.get("name")
        admin = ", ".join([x for x in [g0.get("admin1"), g0.get("country_code")] if x])

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=10,
        ).json()
        cur = wx.get("current") or {}
        t = cur.get("temperature_2m")
        w = cur.get("wind_speed_10m")
        p = cur.get("precipitation")
        code = cur.get("weather_code")
        return f"{place} ({admin}): {t}°C, wind {w} m/s, precip {p} mm, code {code}"
    except Exception as e:
        return f"Weather error: {e}"


@tool
def get_local_city_by_ip() -> str:
    """Fetch the location (city, country) where the agent is running by resolving the local IP address using ip-api.com (no API key)."""

    tool_call_counts["get_local_city_by_ip"] = tool_call_counts.get("get_local_city_by_ip", 0) + 1

    try:
        response = requests.get(f"http://ip-api.com/json/", timeout=5)
        data = response.json()
        result_city = data.get("city", "")
        result_country = data.get("country", "")
        return f"The local IP address is located in {result_city}, {result_country}."
    except Exception as e:
        return f"IP location lookup error: {e}"  # fallback if the request fails

def build_scopus_query(keywords, title, from_year, to_year):
    q = []

    # title field
    if title:
        q.append(f'TITLE("{title}")')

    # keywords list
    if keywords:
        # Combine with AND
        kw = " AND ".join([f'"{k}"' for k in keywords])
        q.append(f'TITLE-ABS-KEY({kw})')

    # year filters
    if from_year:
        q.append(f"PUBYEAR > {from_year - 1}")   # PUBYEAR > 2022 means >= 2023
    if to_year:
        q.append(f"PUBYEAR < {to_year + 1}")

    return " AND ".join(q)


@tool
def scopus_search(
    keywords: list[str],
    title: str = None,
    from_year: int = None,
    to_year: int = None,
):
    """Search SCOPUS for articles by keywords, title, and optional year range. 
    The LLM should only provide the structured parameters, not SCOPUS syntax."""
    
    tool_call_counts["scopus_search"] = tool_call_counts.get("scopus_search", 0) + 1
    #print(f"SCOPUS search called with keywords={keywords}, title={title}, from_year={from_year}, to_year={to_year}")

    # Try to help deepseek or other looping agents by limiting excessive calls:
    if tool_call_counts["scopus_search"] > 30:
        return "Error: SCOPUS_search tool call limit exceeded (30).  Please stop further calls, and provide a final answer based on the results so far."

    # Try assisting deepseek or other looping agents by validating inputs:
    if len(keywords) > 10:
        return "Error: Too many keywords provided. Please limit to 10 or fewer."
    if to_year and to_year > 2026:
        return "Error: 'to_year' cannot be in the future."

    try:
        api_key = os.getenv("SCOPUS_API_KEY")
        query = build_scopus_query(keywords, title, from_year, to_year)

        url = "https://api.elsevier.com/content/search/scopus"
        params = {"query": query, "apiKey": api_key, "count": 10}
        headers = {"Accept": "application/json"}

        r = requests.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"SCOPUS search({keywords}, {title}, {from_year}, {to_year}) failed. Error: {e}")
        return f"SCOPUS search error: {e}"  

@tool
def scopus_get_abstract(doi: str):
    """Get the abstract of a SCOPUS article by DOI."""

    tool_call_counts["scopus_get_abstract"] = tool_call_counts.get("scopus_get_abstract", 0) + 1
    #print(f"SCOPUS get_abstract called with doi={doi}")

    try:
        api_key = os.getenv("SCOPUS_API_KEY")
        url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
        params = {"apiKey": api_key}
        headers = {"Accept": "application/json"}

        r = requests.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"SCOPUS get_abstract({doi}) failed. Error: {e}")
        return f"SCOPUS get_abstract error: {e}"
    
def build_tools() -> List[Any]:
    return [
        weather_now,
        get_local_city_by_ip,
        scopus_search,
        scopus_get_abstract,
    ]

# Define response format
@dataclass
class WeatherResponseFormat:
    """Response schema for the weather agent."""
    descriptive_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

@dataclass
class Citation:
    """Structured citation information."""
    title: str
    doi: Optional[str]
    authors: List[str]
    year: Optional[int]

@dataclass
class SearchResponseFormat:
    """Final structured output from the agent."""
    result_description: str
    citations: Optional[List[Citation]] = None

# -----------------------------
# LLM Builder
# -----------------------------
def build_llm(backend: str, model: Optional[str] = None, temperature: float = 0.0, max_execution_time: int = 60):
    backend = (backend or "openai").lower()

    if backend == "openai":
        name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=name, 
            timeout=max_execution_time,
            temperature=temperature
        )

    elif backend == "gemini":
        # your working Gemini builder here (no convert_system_message_to_human)
        name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("Set GEMINI_API_KEY for --backend gemini.")

        return ChatGoogleGenerativeAI(
            model=name,
            temperature=temperature,
            max_tokens=None,
            timeout=max_execution_time,
            max_retries=2
        )

    elif backend == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") # or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY for --backend deepseek.")
        name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # or "deepseek-reasoner" , "deepseek-coder", etc.
        # return ChatOpenAI(
        #     base_url="https://api.deepseek.com/v1",
        #     model="deepseek-chat",
        #     api_key=api_key,
        #     temperature=0.1,)

        return ChatDeepSeek(
            model=name, 
            timeout=max_execution_time,
            temperature=temperature,
            top_p=0.2,
            max_retries=2
        )

    elif backend == "ollama":
        def get_active_ollama_url():
            candidate_urls = [
                "http://10.132.11.9:11434",         # GCP Simula server; works from the lab and via VPN
                "http://localhost:11434",           # Inside container? No, use host.docker.internal
                "http://host.docker.internal:11434", # Docker Desktop (Mac/Windows/Linux 20.10+)
                "http://host-gateway:11434",        # Linux Docker 20.10+
                "http://172.17.0.1:11434"           # Docker bridge gateway (fallback)
                ]
            
            for url in candidate_urls:
                try:
                    response = requests.head(f"{url}/api/tags", timeout=3)
                    if response.status_code == 200:
                        print(f"Ollama active at: {url}")
                        return url
                except requests.RequestException:
                    continue
            
            raise ConnectionError("No active Ollama server found")

        llm = ChatOllama(
            model=model or "llama3.1", # Others tested "qwen3:8b", "mistral"
            base_url = get_active_ollama_url(),
            temperature=temperature,
            validate_model_on_init=True,
            num_predict=-2,
            disable_streaming=True,
        )
        return llm
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# -----------------------------
# Agent
# -----------------------------

def create_system_prompt(agent_type: str, tools) -> str:
    if agent_type == "weather":
        system_prompt = f"""You are a helpful AI weather agent that can use tools.
        Available tools:
        {tools}
        Always use the tools to complete the described action. Follow ReAct strictly.
        """
    elif agent_type == "research-assistant":
        system_prompt = f"""You are a helpful AI research assistant with access to SCOPUS database and that can use tools.
        Available tools:
        {tools}
        You can use the tools repeatedly to gather information, retrieve abstracts, and reason in order to answer user queries about scientific literature.        
        Always use the tools to complete the described action. Follow ReAct strictly.
        When in doubt, reason first, then act.
        """
        #Provide final answers concisely when you have sufficient information
        #After each tool call, analyze the results before deciding next steps
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    #print("System prompt:", system_prompt)
    return system_prompt

def build_agent(llm, tools, agent_type: str, system_prompt):
    response_format: Union[ToolStrategy, None]
    if agent_type == "weather":
        response_format=ToolStrategy(WeatherResponseFormat)
    elif agent_type == "research-assistant":
        response_format=ToolStrategy(SearchResponseFormat)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    agent = create_agent(
        model=llm,
        system_prompt=system_prompt,
        tools=tools,
        response_format=response_format,
    )
    return agent


def extract_final_ai_message(messages):
    """
    Given a list of LangChain message objects, return the content
    of the last AIMessage that contains non-empty content.
    """
    final_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            final_message = msg.content.strip()
    return final_message  # or raise an error if desired

# -----------------------------
# Execution
# -----------------------------
def run_query(agent, prompt: str, max_execution_time: int) -> str:
    _config = {
    "timeout": str(max_execution_time),  # Timeout in seconds
}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        #config=_config,
    )

    out = None
    try:
        # This works on standard LangChain models like OpenAI/Gemini
        out = response["structured_response"] 
    except Exception as e:
        # Desperate fallback: try to extract the final AI message content
        out = str(extract_final_ai_message(response.get("messages", []))) or str(response)
    return out

# -----------------------------
# CLI
# -----------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ReAct Agent — Reasoning and tool use")
    parser.add_argument("--agent", type=str, default="weather", 
                        help="Type of agent to run",
                        choices=["weather", "research-assistant"])
    parser.add_argument("--backend", type=str, default="openai",
                    choices=["openai", "gemini", "deepseek", "ollama"],
                    help="LLM backend")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-execution-time", type=int, default=60)
    parser.add_argument("--pcap-dir", type=str, default=PCAP_DIR, help="Directory to save PCAP files")
    parser.add_argument("--no-wikipedia", action="store_true")
    parser.add_argument("--ddg", action="store_true")
    parser.add_argument("--serpapi", action="store_true")
    args = parser.parse_args()

    # ---------- Add some reasonable defaults ----------
    if args.agent == "weather":
        if not args.prompt:
            args.prompt = "What is the current weather in New York City, USA?"
    elif args.agent == "research-assistant":
        if not args.prompt:
            args.prompt = "Retrieve three IoT papers with citations, which are published after 2022 and talk about privacy in smart homes. Explain."
        # We normally use relatively modest models.  Research-assistant may need stronger ones.
        if not args.model:
            if args.backend == "openai":
                args.model = "gpt-5"  # or "gpt-4o"
            elif args.backend == "gemini":
                args.model = "gemini-2.5-pro"
            elif args.backend == "deepseek":
                args.model = "deepseek-chat" # "deepseek-reasoner" does not support tool calling :(
            elif args.backend == "ollama":
                args.model = "qwen3:32b"

    # ---------- PCAP Capture ----------
    sniffer = None
    os.makedirs(PCAP_DIR, exist_ok=True)
    print("\n Starting packet capture (may require sudo/root)...")
    try:
        sniffer = AsyncSniffer(store=True)
        sniffer.start()
        # tiny sleep to ensure sniffer is up before the LLM call
        time.sleep(1)
        print("Packet capture started.")
    except Exception as e:
        print(f"Failed to start packet capture: {e}")
        sniffer = None

    tools = build_tools()
    system_prompt = create_system_prompt(args.agent, tools)
    llm = build_llm(backend=args.backend, model=args.model, temperature=args.temperature, max_execution_time=args.max_execution_time)
    agent = build_agent(llm, tools, agent_type=args.agent, system_prompt=system_prompt)

       # ---------- Agent run ----------
    run_dir = None
    logger = None

    try:
        print("\n=== ReAct Agent Run ===")
        print("Start Time:", datetime.now(timezone.utc).astimezone().isoformat())
        print(f"Agent: {args.agent}")
        print(f"Prompt: {args.prompt}\n")
        print(f"Using backend: {args.backend}, model: {args.model or 'default'}, temperature: {args.temperature}")
        output = run_query(agent, args.prompt, args.max_execution_time)
        print(f"Query completed. Tool call counts: {tool_call_counts}")
        print("\n--- Agent Output ---\n")
        print(output)
        print("\n=== End of Run ===")
        print("End Time:", datetime.now(timezone.utc).astimezone().isoformat()) 
        
    finally:
         # ---------- Stop & save PCAP ----------
        if sniffer and getattr(sniffer, "running", False):
            print("\n Stopping packet capture...")
            try:
                sniffer.stop()
                ts = time.strftime("%Y%m%d-%H%M%S")
                backend = getattr(args, "backend", "unknown")
                model = args.model or "default"
                safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
                pcap_filename = os.path.join(
                    args.pcap_dir, f"{backend}-{safe_model}-{ts}.pcap"
                )
                scapy.wrpcap(pcap_filename, sniffer.results)
                print(f" Packets saved to '{pcap_filename}'")
            except Exception as e:
                print(f"Failed to save packet capture: {e}")

if __name__ == "__main__":
    main()
