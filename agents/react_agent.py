"""
ReAct Agent — clean compatibility build with Google Search + live weather
- Works on Python 3.10–3.13
- Supports modern LangChain (≥0.2) and falls back to legacy (<0.2)
- Tools included:  optional Google (SerpAPI), optional DuckDuckGo, optional Wikipedia, and `weather_now` (Open‑Meteo, no API key)
- Backends supported: OpenAI (ChatGPT), Gemini (Google AI Studio), DeepSeek, LLama
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from huggingface_hub import InferenceClient
import unicodedata 
import re
from langchain_core.prompts import ChatPromptTemplate

import requests
from dotenv import load_dotenv

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
# LLM backends (OpenAI, Gemini)
# -----------------------------
try:
    from langchain_openai import ChatOpenAI
except Exception:
    from langchain.chat_models import ChatOpenAI  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # pip install langchain-google-genai google-generativeai
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

# -----------------------------
# Agent APIs 
# -----------------------------
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool



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

def build_tools() -> List[Tool]:
    return [
        Tool(name="weather_now", func=weather_now, description="Get current weather for a city, e.g., 'London'"),
    ]


# ---------------------------------------------------------
# LLAMA (Hugging Face Inference API)
# ---------------------------------------------------------
# --- HF Wrapper for LLaMA (Hugging Face Inference API) ---
from typing import Any, List
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage
)


from typing import Any, List
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage
)

class HFLLMWrapper:
    """
    Wrapper so HF models behave like LangChain chat models.
    """

    def __init__(self, client, model_name: str, temperature: float = 0.0):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert LangChain messages (or tuples) → text prompt."""
        parts = []
        for m in messages:
            # Normal LangChain message types
            if isinstance(m, SystemMessage):
                parts.append(f"<system>\n{m.content}\n</system>")
            elif isinstance(m, HumanMessage):
                parts.append(f"<human>\n{m.content}\n</human>")
            elif isinstance(m, AIMessage):
                parts.append(f"<assistant>\n{m.content}\n</assistant>")

            # Sometimes LangChain passes ('human', 'text') or similar
            elif isinstance(m, tuple) and len(m) == 2:
                role, content = m
                role = str(role).lower()
                if role in ("human", "user"):
                    parts.append(f"<human>\n{content}\n</human>")
                elif role in ("assistant", "ai"):
                    parts.append(f"<assistant>\n{content}\n</assistant>")
                elif role == "system":
                    parts.append(f"<system>\n{content}\n</system>")
                else:
                    parts.append(str(m))

            else:
                # Fallback: just stringize whatever this object is
                parts.append(str(m))

        return "\n".join(parts)

    def invoke(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        prompt = self._messages_to_prompt(messages)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=self.temperature,
        )

        out = resp.choices[0].message["content"]
        return AIMessage(content=out)

    # LangChain compatibility — allow llm(messages)
    def __call__(self, messages: List[Any], **kwargs):
        return self.invoke(messages, **kwargs)

    # Critical: allow create_react_agent() to call llm.bind(...)
    def bind(self, **kwargs):
        # You can store/use kwargs if you want, but returning self is enough here.
        return self



def load_llama(model_name: str, temperature: float = 0.2):
    """
    Loads a LLaMA model from HuggingFace using the Inference API.
    This returns a LangChain-compatible LLM wrapper.
    """

    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        raise ValueError("HF_API_KEY is not set!")

    # Hugging Face inference client
    client = InferenceClient(
        model=model_name,
        token=hf_key,
    )

    # LangChain wrapper – unify with your existing architecture
    class HFLLMWrapper:
        def __init__(self, client, model_name, temperature):
            self.client = client
            self.model_name = model_name
            self.temperature = temperature

        def invoke(self, messages, **kwargs):
            # Convert messages → chat format
            prompt = ""
            for m in messages:
                if m.type == "system":
                    prompt += f"<system>{m.content}</system>\n"
                elif m.type == "human":
                    prompt += f"<human>{m.content}</human>\n"
                elif m.type == "assistant":
                    prompt += f"<assistant>{m.content}</assistant>\n"

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=self.temperature,
                stream=False
            )

            return response.choices[0].message["content"]

    return HFLLMWrapper(client, model_name, temperature)

# -----------------------------
# LLM Builder
# -----------------------------
def build_llm(model: Optional[str] = None, temperature: float = 0.0, backend: str = "openai"):
    backend = (backend or "openai").lower()

    if backend == "openai":
        name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=name, temperature=temperature)

    elif backend == "gemini":
        # your working Gemini builder here (no convert_system_message_to_human)
        name = model or os.getenv("GEMINI_MODEL")
        return _make_gemini_llm(name, temperature)

    elif backend == "deepseek":
        # DeepSeek is OpenAI-compatible; just change base_url + key.
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY for --backend deepseek.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # or "deepseek-reasoner"
        # Note: Some deployments use /v1; both usually work:
        return ChatOpenAI(model=name, temperature=temperature, api_key=api_key, base_url=base_url)

    elif backend == "llama":
        from huggingface_hub import InferenceClient
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise RuntimeError("Set HF_API_KEY for --backend llama.")
        client = InferenceClient(model=model, token=hf_key)
        return HFLLMWrapper(client, model_name=model, temperature=temperature)
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")



def _make_gemini_llm(requested: Optional[str], temperature: float):
    """
    Robust Gemini constructor: try a list of known IDs and fall back until one works.
    Handles per-account availability differences that cause 404s.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    candidates = [c for c in [
        requested,
        os.getenv("GEMINI_MODEL"),
        # Prefer newest names first; keep a few older fallbacks:
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ] if c]

    last_err = None
    for mid in candidates:
        try:
            return ChatGoogleGenerativeAI(model=mid, temperature=temperature)
        except Exception as e:
            # If it's a NotFound / 404, try next candidate
            last_err = e
            continue
    raise RuntimeError(
        "No Gemini model from the candidate list is available for this API key. "
        "Try setting --model explicitly to one returned by your account’s ListModels."
    ) from last_err



# -----------------------------
# Agent
# -----------------------------

def build_agent(llm, tools, backend: str = "openai"):
    # Force the LLM to stop right after a tool call so the parser can insert Observation:
    llm_for_agent = llm.bind(stop=["\nObservation:"])

    # Gemini crashes if we add a second SystemMessage later in the chat.
    # Use HUMAN scratchpad for Gemini; SYSTEM scratchpad for OpenAI.
    scratchpad_role = "human" if backend.lower() == "gemini" else "system"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI that can use tools.\n"
         "Available tools:\n{tools}\n"
         "You may call tools from this set: {tool_names}.\n"
         "Follow ReAct strictly. Valid prefixes: Thought:, Action:, Action Input:, Observation:, Final Answer:\n"
         "\n"
         "TOOL-CALL FORMAT (exactly, no extra words):\n"
         "Action: <tool_name>\n"
         "Action Input: \"<JSON-serializable string>\"\n"
         "Then STOP and wait for Observation.\n"
         "After Observation, finish with:\n"
         "Final Answer: <your answer>\n"
        ),
        ("human", "{input}"),
        (scratchpad_role, "{agent_scratchpad}"),
    ])

    runnable = create_react_agent(llm_for_agent, tools, prompt)
    return AgentExecutor(
        agent=runnable,
        tools=tools,
        verbose=True,
        max_iterations=20,
        max_execution_time=120,
        handle_parsing_errors=True,
    )


# -----------------------------
# Execution
# -----------------------------
def run_query(agent, prompt: str) -> str:
    out = agent.invoke({"input": prompt})
    return out.get("output") if isinstance(out, dict) else str(out)

# -----------------------------
# CLI
# -----------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ReAct Agent — Google Search + live weather (OpenAI/Gemini)")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--backend", type=str, default="openai",
                    choices=["openai", "gemini", "deepseek", "llama"],
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
        tools = build_tools()
        llm = build_llm(model=args.model, temperature=args.temperature, backend=args.backend)
        agent = build_agent(llm, tools, backend=args.backend)

        print("\n=== ReAct Agent Run ===")
        print(f"Prompt: {args.prompt}\n")
        output = run_query(agent, args.prompt)
        print("\n--- Agent Output ---\n")
        print(output)
        
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
                    PCAP_DIR, f"{backend}-{safe_model}-{ts}.pcap"
                )
                scapy.wrpcap(pcap_filename, sniffer.results)
                print(f" Packets saved to '{pcap_filename}'")
            except Exception as e:
                print(f"Failed to save packet capture: {e}")
        elif SCAPY_AVAILABLE and not sniffer:
            print(" Packet capture was not active.")


if __name__ == "__main__":
    main()
