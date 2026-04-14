from dotenv import load_dotenv
import argparse
import os
import time
from datetime import datetime
from typing import Any, List, Optional
from dataclasses import dataclass
import json
import re
import unicodedata

import requests
import scapy.all as scapy
from scapy.sendrecv import AsyncSniffer

from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage

PCAP_DIR = "./pcap/planner_executor/"

# -----------------------------
# Tool call counters
# -----------------------------
tool_call_counts = {}


def _clean_city(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r'^[\s\'"]+|[\s\'"]+$', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(", UK", ", United Kingdom")
    s = s.replace(", U.K.", ", United Kingdom")
    return s


@tool
def weather_now(city: str) -> str:
    """Fetch current weather for a city using Open-Meteo (no API key)."""
    tool_call_counts["weather_now"] = tool_call_counts.get("weather_now", 0) + 1

    try:
        q = _clean_city(city)
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
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        result_city = data.get("city", "")
        result_country = data.get("country", "")
        return f"The local IP address is located in {result_city}, {result_country}."
    except Exception as e:
        return f"IP location lookup error: {e}"



def build_scopus_query(keywords, title, from_year, to_year):
    q = []
    if title:
        q.append(f'TITLE("{title}")')
    if keywords:
        kw = " AND ".join([f'"{k}"' for k in keywords])
        q.append(f"TITLE-ABS-KEY({kw})")
    if from_year:
        q.append(f"PUBYEAR > {from_year - 1}")
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
    """Search SCOPUS for articles by keywords, title, and optional year range."""
    tool_call_counts["scopus_search"] = tool_call_counts.get("scopus_search", 0) + 1

    if tool_call_counts["scopus_search"] > 30:
        return "Error: SCOPUS_search tool call limit exceeded (30). Please stop further calls and provide a final answer based on the results so far."
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


# -----------------------------
# Structured outputs
# -----------------------------
@dataclass
class PlanStep:
    id: int
    description: str


@dataclass
class PlannerPlan:
    task_type: str
    steps: List[PlanStep]
    final_goal: str


@dataclass
class ExecutorResult:
    step_id: int
    step_description: str
    result: str
    status: str


@dataclass
class FinalAnswer:
    answer: str


# -----------------------------
# LLM Builder
# -----------------------------
def build_llm(backend: str, model: Optional[str] = None, temperature: float = 0.0, max_execution_time: int = 60):
    backend = (backend or "openai").lower()

    if backend == "openai":
        name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=name, timeout=max_execution_time, temperature=temperature)

    elif backend == "gemini":
        name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("Set GEMINI_API_KEY for --backend gemini.")
        return ChatGoogleGenerativeAI(
            model=name,
            temperature=temperature,
            max_tokens=None,
            timeout=max_execution_time,
            max_retries=2,
        )

    elif backend == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY for --backend deepseek.")
        name = model or "deepseek-chat"
        return ChatOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model=name,
            temperature=temperature,
            timeout=max_execution_time,
        )

    elif backend == "ollama":
        def get_active_ollama_url():
            candidate_urls = [
                "http://10.132.11.9:11434",
                "http://localhost:11434",
                "http://host.docker.internal:11434",
                "http://host-gateway:11434",
                "http://172.17.0.1:11434",
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
            model=model or "llama3.1",
            base_url=get_active_ollama_url(),
            temperature=temperature,
            validate_model_on_init=True,
            num_predict=-2,
            disable_streaming=True,
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")


# -----------------------------
# Prompts
# -----------------------------
def create_planner_system_prompt() -> str:
    return """
You are the PLANNER agent in a planner-executor architecture.
Your responsibilities:
- Understand the user's overall task.
- Break it into a short ordered list of executable steps.
- Keep steps concrete and tool-oriented.
- Do not execute tools yourself.
- Do not solve the task directly.

Rules:
- Produce 1 to 4 steps only.
- Each step must be self-contained and executable by another agent.
- Use simple wording.
- For weather tasks, typical steps are: identify location, get weather, interpret and recommend activities.
- For research tasks, typical steps are: formulate search, search literature, inspect abstracts, summarize.
""".strip()



def create_executor_system_prompt(tools) -> str:
    return f"""
You are the EXECUTOR agent in a planner-executor architecture.
Your responsibilities:
- Execute exactly one assigned step.
- Use tools when needed.
- Return the result for that step only.
- Do not redesign the overall workflow.
- Do not claim you completed the full user task unless the assigned step explicitly asks for final synthesis.

Available tools:
{tools}

Rules:
- Prefer tool use when factual lookup is needed.
- Be concise but complete.
- If a step cannot be completed, explain why.
""".strip()



def create_synthesizer_system_prompt() -> str:
    return """
You are the FINAL SYNTHESIZER in a planner-executor architecture.
You receive the user's original prompt plus the completed step results.
Your job is to write the final answer for the user.
Do not invent tool results that are not present in the execution notes.
Be concise and directly answer the user's question.
""".strip()


# -----------------------------
# Agent builders
# -----------------------------
def build_planner_agent(llm):
    return create_agent(
        model=llm,
        system_prompt=create_planner_system_prompt(),
        tools=[],
        response_format=ToolStrategy(PlannerPlan),
    )



def build_executor_agent(llm, tools):
    return create_agent(
        model=llm,
        system_prompt=create_executor_system_prompt(tools),
        tools=tools,
        response_format=ToolStrategy(ExecutorResult),
    )



def build_synthesizer_agent(llm):
    return create_agent(
        model=llm,
        system_prompt=create_synthesizer_system_prompt(),
        tools=[],
        response_format=ToolStrategy(FinalAnswer),
    )


# -----------------------------
# Helpers
# -----------------------------
def extract_final_ai_message(messages):
    final_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                final_message = content
                break
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if text:
                            final_message = text
                            break
                if final_message:
                    break
    return final_message



def invoke_with_fallback(agent, prompt: str):
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    try:
        return response["structured_response"]
    except Exception:
        return extract_final_ai_message(response.get("messages", [])) or str(response)



def stringify_plan(plan: PlannerPlan) -> str:
    lines = [f"Task type: {plan.task_type}", f"Final goal: {plan.final_goal}"]
    for step in plan.steps:
        lines.append(f"Step {step.id}: {step.description}")
    return "\n".join(lines)


# -----------------------------
# Planner-executor run
# -----------------------------
def run_planner_executor(planner, executor, synthesizer, prompt: str):
    plan = invoke_with_fallback(planner, prompt)
    if not isinstance(plan, PlannerPlan):
        raise RuntimeError(f"Planner did not return PlannerPlan. Got: {plan}")

    print("\n--- Planner Output ---\n")
    print(stringify_plan(plan))

    execution_results: List[ExecutorResult] = []

    for step in plan.steps:
        executor_prompt = f"""
Original user task:
{prompt}

Overall plan:
{stringify_plan(plan)}

Assigned step:
Step {step.id}: {step.description}

Execute only this step and return the result.
""".strip()

        result = invoke_with_fallback(executor, executor_prompt)
        if not isinstance(result, ExecutorResult):
            result = ExecutorResult(
                step_id=step.id,
                step_description=step.description,
                result=str(result),
                status="completed",
            )

        execution_results.append(result)
        print("\n--- Executor Output ---\n")
        print(f"Step {result.step_id}: {result.step_description}")
        print(f"Status: {result.status}")
        print(f"Result: {result.result}")

    notes = []
    for item in execution_results:
        notes.append(
            f"Step {item.step_id} ({item.step_description}) -> [{item.status}] {item.result}"
        )

    synth_prompt = f"""
Original user task:
{prompt}

Planner output:
{stringify_plan(plan)}

Execution notes:
{chr(10).join(notes)}

Write the final answer to the user.
""".strip()

    final = invoke_with_fallback(synthesizer, synth_prompt)
    if isinstance(final, FinalAnswer):
        return final.answer
    return str(final)


# -----------------------------
# CLI
# -----------------------------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Planner-Executor multi-agent demo")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "gemini", "deepseek", "ollama"], help="LLM backend")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-execution-time", type=int, default=60)
    parser.add_argument("--pcap-dir", type=str, default=PCAP_DIR, help="Directory to save PCAP files")
    args = parser.parse_args()

    if not args.prompt:
        args.prompt = "What is the current weather in London, UK, and what activities do you recommend?"

    if not args.model:
        if args.backend == "openai":
            args.model = "gpt-4o-mini"
        elif args.backend == "gemini":
            args.model = "gemini-2.5-flash"
        elif args.backend == "deepseek":
            args.model = "deepseek-chat"
        elif args.backend == "ollama":
            args.model = "llama3.1"
# -----------------------------
# PCAP Capture
# -----------------------------
    sniffer = None
    os.makedirs(args.pcap_dir, exist_ok=True)
    print("\nStarting packet capture (may require sudo/root)...")
    try:
        sniffer = AsyncSniffer(store=True)
        sniffer.start()
        time.sleep(1)
        print("Packet capture started.")
    except Exception as e:
        print(f"Failed to start packet capture: {e}")
        sniffer = None

    tools = build_tools()
    llm = build_llm(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_execution_time=args.max_execution_time,
    )

    planner = build_planner_agent(llm)
    executor = build_executor_agent(llm, tools)
    synthesizer = build_synthesizer_agent(llm)

    try:
        print("\n=== Planner-Executor Agent Run ===")
        print("Start Time:", datetime.now().astimezone().isoformat())
        print(f"Prompt: {args.prompt}\n")
        print(f"Using backend: {args.backend}, model: {args.model}, temperature: {args.temperature}")

        output = run_planner_executor(planner, executor, synthesizer, args.prompt)

        print(f"\nQuery completed. Tool call counts: {tool_call_counts}")
        print("\n--- Final Output ---\n")
        print(output)
        print("\n=== End of Run ===")
        print("End Time:", datetime.now().astimezone().isoformat())

    finally:
        if sniffer and getattr(sniffer, "running", False):
            print("\nStopping packet capture...")
            try:
                sniffer.stop()
                ts = time.strftime("%Y%m%d-%H%M%S")
                backend = getattr(args, "backend", "unknown")
                model = args.model or "default"
                safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
                pcap_filename = os.path.join(args.pcap_dir, f"{backend}-{safe_model}-{ts}.pcap")
                scapy.wrpcap(pcap_filename, sniffer.results)
                print(f"Packets saved to '{pcap_filename}'")
            except Exception as e:
                print(f"Failed to save packet capture: {e}")


if __name__ == "__main__":
    main()
