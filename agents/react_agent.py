"""
ReAct Agent — clean compatibility build with Google Search + live weather
- Works on Python 3.10–3.13
- Supports modern LangChain (≥1.1)
- Tools included:  optional Google (SerpAPI), optional DuckDuckGo, optional Wikipedia, and `weather_now` (Open‑Meteo, no API key)
- Backends supported: OpenAI (ChatGPT), Gemini (Google AI Studio), DeepSeek, LLama
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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

try:
    import scapy.all as scapy
    from scapy.sendrecv import AsyncSniffer
    SCAPY_AVAILABLE = True
    PCAP_DIR = "./pcap/react/"
except (ImportError, OSError) as e:
    print(f"Warning: scapy unavailable. Packet capture disabled. {e}")
    SCAPY_AVAILABLE = False
    PCAP_DIR = "./pcap/react/"


# -----------------------------
# Agent APIs 
# -----------------------------


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
    try:
        response = requests.get(f"http://ip-api.com/json/", timeout=5)
        data = response.json()
        result_city = data.get("city", "")
        result_country = data.get("country", "")
        return f"The local IP address is located in {result_city}, {result_country}."
    except Exception as e:
        return f"IP location lookup error: {e}"  # fallback if the request fails

def build_tools() -> List[Any]:
    return [
        #Tool(name="weather_now", func=weather_now, description="Get current weather for a city, e.g., 'London'"),
        weather_now,
        get_local_city_by_ip,
    ]

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    descriptive_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# -----------------------------
# LLM Builder
# -----------------------------
def build_llm(model: Optional[str] = None, temperature: float = 0.0, max_execution_time: int = 60, backend: str = "openai"):
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
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY for --backend deepseek.")
        name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # or "deepseek-reasoner" , "deepseek-coder", etc.
        return ChatDeepSeek(
            model=name, 
            timeout=max_execution_time,
            temperature=temperature
        )

    elif backend == "ollama":
        def get_active_ollama_url():
            candidate_urls = [
            "http://localhost:11434",           # Inside container? No, use host.docker.internal
            "http://host.docker.internal:11434", # Docker Desktop (Mac/Windows/Linux 20.10+)
            "http://host-gateway:11434",         # Linux Docker 20.10+
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

        return ChatOllama(
            model=model or "llama3.1", # Others tested "qwen3:8b", "mistral"
            base_url = get_active_ollama_url(),
            temperature=temperature,
            validate_model_on_init=True,
            num_predict=256,
            max_retries=2,
            timeout=max_execution_time,
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# -----------------------------
# Agent
# -----------------------------

def build_agent(llm, tools, backend: str = "openai"):
    # Force the LLM to stop right after a tool call so the parser can insert Observation:
    #llm_for_agent = llm.bind(stop=["\nObservation:"])
    
    # Gemini crashes if we add a second SystemMessage later in the chat.
    # Use HUMAN scratchpad for Gemini; SYSTEM scratchpad for OpenAI.
    scratchpad_role = "human" if backend.lower() == "gemini" else "system"

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "You are a helpful AI that can use tools.\n"
    #      "Available tools:\n{tools}\n"
    #      "You may call tools from this set: {tool_names}.\n"
    #      "Follow ReAct strictly. Valid prefixes: Thought:, Action:, Action Input:, Observation:, Final Answer:\n"
    #      "\n"
    #      "TOOL-CALL FORMAT (exactly, no extra words):\n"
    #      "Action: <tool_name>\n"
    #      "Action Input: \"<JSON-serializable string>\"\n"
    #      "Then STOP and wait for Observation.\n"
    #      "After Observation, finish with:\n"
    #      "Final Answer: <your answer>\n"
    #     ),
    #     ("human", "{input}"),
    #     (scratchpad_role, "{agent_scratchpad}"),
    # ])

    # Expand tools into a human-readable list for the system prompt
    tool_entries: List[str] = []
    tool_names: List[str] = []
    for t in tools:
        # LangChain Tool objects typically have .name and .description; fallback to function attributes
        name = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
        desc = getattr(t, "description", None) or getattr(t, "__doc__", None) or ""
        name = str(name)
        desc = str(desc).strip()
        tool_entries.append(f"{name}: {desc}" if desc else f"{name}")
        tool_names.append(name)

    tools_text = "\n".join(f"  - {entry}" for entry in tool_entries) if tool_entries else "  (no tools available)"
    tool_names_text = ", ".join(tool_names) if tool_names else "(none)"

    SYSTEM_PROMPT = f"""You are a helpful AI that can use tools.
        Available tools:
        {tools}
        Always use the tools to complete the described action. Follow ReAct strictly.
        """
        # If no city name is provided in the prompt, use 'get_local_city_by_ip' to find out the city and country where you are running.
        # Always use 'weather_now' to get the current weather. 
        # Follow ReAct strictly.
        # You may call tools from this set: {tool_names_text}.
        # Follow ReAct strictly. Valid prefixes: Thought:, Action:, Action Input:, Observation:, Final Answer:

        # TOOL-CALL FORMAT (exactly, no extra words):
        # Action: <tool_name>
        # Action Input: \"<JSON-serializable string>\"
        # Then STOP and wait for Observation.
        # After Observation, finish with:
        # Final Answer: <your answer>
    #print(SYSTEM_PROMPT)

    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        response_format=ToolStrategy(ResponseFormat),
    )
    return agent


from langchain_core.messages import AIMessage
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
    #out = response.get("messages") if isinstance(response, dict) else str(response)
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
    parser = argparse.ArgumentParser(description="ReAct Agent — Google Search + live weather (OpenAI/Gemini)")
    parser.add_argument("--prompt", type=str, default="What is the current weather in London, UK?")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-execution-time", type=int, default=60)
    parser.add_argument("--pcap-dir", type=str, default=PCAP_DIR, help="Directory to save PCAP files")
    parser.add_argument("--backend", type=str, default="openai",
                    choices=["openai", "gemini", "deepseek", "ollama"],
                    help="LLM backend")
    parser.add_argument("--no-wikipedia", action="store_true")
    parser.add_argument("--ddg", action="store_true")
    parser.add_argument("--serpapi", action="store_true")
    args = parser.parse_args()

    # ---------- PCAP Capture ----------
    sniffer = None
    if SCAPY_AVAILABLE:
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
    else:
        print("Packet capture disabled (scapy not available).")

    tools = build_tools()
    llm = build_llm(model=args.model, temperature=args.temperature, backend=args.backend)
    agent = build_agent(llm, tools,backend=args.backend)

       # ---------- Agent run ----------
    run_dir = None
    logger = None

    try:
        print("\n=== ReAct Agent Run ===")
        print("Start Time:", datetime.now(timezone.utc).astimezone().isoformat())
        print(f"Prompt: {args.prompt}\n")
        print(f"Using backend: {args.backend}, model: {args.model or 'default'}, temperature: {args.temperature}")
        output = run_query(agent, args.prompt, args.max_execution_time)
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
        elif SCAPY_AVAILABLE and not sniffer:
            print(" Packet capture was not active.")


if __name__ == "__main__":
    main()
