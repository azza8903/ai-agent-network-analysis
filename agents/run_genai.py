import os
from google import genai
from google.genai import types
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

# --- Optional Packet Capture with scapy ---
try:
    import scapy.all as scapy
    from scapy.sendrecv import AsyncSniffer
    SCAPY_AVAILABLE = True
    PCAP_DIR = "./pcap/genai/"
except (ImportError, OSError) as e:
    print(f"Warning: scapy unavailable. Packet capture disabled. {e}")
    SCAPY_AVAILABLE = False

def main(prompt: str, model_id: str):
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

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    print("\n🧠 Running agent task...")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )

    except Exception as e:
        print(f"❌ Error running agent: {e}")
        response = None
    finally:
        if sniffer and sniffer.running:
            print("\n🛑 Stopping packet capture...")
            try:
                sniffer.stop()
                PCAP_FILENAME = f"{PCAP_DIR}{model_id}-{time.strftime('%Y%m%d-%H%M%S')}.pcap"
                scapy.wrpcap(PCAP_FILENAME, sniffer.results)
                print(f"📦 Packets saved to '{PCAP_FILENAME}'")
            except Exception as e:
                print(f"⚠️ Failed to save packet capture: {e}")
        elif SCAPY_AVAILABLE and not sniffer:
            print("⚠️ Packet capture was not active.")

    return response


# --- CLI Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a tool-using agent with optional packet capture.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model ID to use (default: gemini-2.5-flash)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Please get the current weather for London and recommend activities based on the conditions",
        help="Prompt for the agent to process (default: 'Please get the current weather for London and recommend activities based on the conditions')"
    )

    args = parser.parse_args()

    response = main(args.prompt, args.model_id)
    if response:
        print(response.text)
    else:
        print("No response received.")
