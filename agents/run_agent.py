import os
import re
import time
import argparse
import requests
import warnings

from dotenv import load_dotenv
from markdownify import markdownify
from requests.exceptions import RequestException

# --- Optional Packet Capture with scapy ---
try:
    import scapy.all as scapy
    from scapy.sendrecv import AsyncSniffer
    SCAPY_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: scapy unavailable. Packet capture disabled. {e}")
    SCAPY_AVAILABLE = False

# --- smolagents Components ---
from smolagents import tool
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    DuckDuckGoSearchTool,
)

# --- Load Environment Variables ---
load_dotenv()
PCAP_FILENAME = "agent_traffic.pcap"

# --- Authenticate Hugging Face ---
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    try:
        login(hf_token)
        print("✅ Logged into Hugging Face Hub.")
    except Exception as e:
        print(f"⚠️ Hugging Face login failed: {e}")
else:
    print("⚠️ HF_TOKEN not set. Gated models may not work.")

# --- Custom Tool: Visit Webpage ---
@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given URL and returns its content as Markdown.

    Args:
        url (str): The URL of the webpage to visit.

    Returns:
        str: The content of the webpage converted to Markdown, or an
             error message if the request fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible)'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        return re.sub(r"\n{3,}", "\n\n", markdown_content)
    except RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# --- Main Agent Execution ---
def main(model_id: str, provider: str):
    print(f"\n🚀 Initializing model: {model_id} via provider: {provider}")
    model = InferenceClientModel(model_id=model_id, provider=provider)

    web_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), visit_webpage],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Performs web searches and visits pages.",
        verbosity_level=1,
    )

    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"],
        verbosity_level=1,
    )

    task = "Please get the current weather for London and recommend activities based on the conditions"
    sniffer = None

    if SCAPY_AVAILABLE:
        print("\n📡 Starting packet capture (requires sudo/root)...")
        try:
            sniffer = AsyncSniffer(store=True)
            sniffer.start()
            time.sleep(1)
            print("✅ Packet capture started.")
        except Exception as e:
            print(f"⚠️ Failed to start packet capture: {e}")
            sniffer = None
    else:
        print("ℹ️ Packet capture disabled.")

    print("\n🧠 Running agent task...")
    try:
        final_answer = manager_agent.run(task)
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        final_answer = None
    finally:
        if sniffer and sniffer.running:
            print("\n🛑 Stopping packet capture...")
            try:
                sniffer.stop()
                scapy.wrpcap(PCAP_FILENAME, sniffer.results)
                print(f"📦 Packets saved to '{PCAP_FILENAME}'")
            except Exception as e:
                print(f"⚠️ Failed to save packet capture: {e}")
        elif SCAPY_AVAILABLE and not sniffer:
            print("⚠️ Packet capture was not active.")

    print("\n✅ Final Answer:")
    print(final_answer or "No output returned.")

# --- CLI Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a tool-using agent with optional packet capture.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="Hugging Face model ID to use (default: Qwen/Qwen2.5-Coder-32B-Instructt)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="together",
        help="Provider to use for the model (default: together)"
    )
    args = parser.parse_args()
    main(args.model_id, args.provider)
