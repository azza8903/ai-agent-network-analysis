import os
import time
import argparse
from dotenv import load_dotenv
import ai_agent_manager as aiam

load_dotenv()

# --- Optional Packet Capture with scapy ---
try:
    import scapy.all as scapy
    from scapy.sendrecv import AsyncSniffer
    SCAPY_AVAILABLE = True
    PCAP_DIR = "./pcap/"
except (ImportError, OSError) as e:
    print(f"Warning: scapy unavailable. Packet capture disabled. {e}")
    SCAPY_AVAILABLE = False

def main(agent: aiam.AI_agent_manager, prompt: str):
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
        response = agent.run(prompt)
        print("✅ Agent task completed.")

    except Exception as e:
        print(f"❌ Error running agent: {e}")
        response = None
    finally:
        if sniffer and sniffer.running:
            print("\n🛑 Stopping packet capture...")
            try:
                sniffer.stop()
                PCAP_FILENAME = f"{PCAP_DIR}{agent.get_name()}-{time.strftime('%Y%m%d-%H%M%S')}.pcap"
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

    # This is the default model for default "gemini" agent.  If any other agent is used, this has to be changed
    # to the default for that agent type, if the user did not set --model-id in the command line.
    DEF_MODEL_ID = "gemini-2.5-flash" 

    parser.add_argument(
        "--agent",
        type=str,
        default="gemini",
        help="AI model ID to use (default: gemini, supported: smal)"
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=DEF_MODEL_ID,
        help="Model ID to use (default: gemini-2.5-flash)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="together",
        help="Provider to use for the smalagents model (default: together)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Please get the current weather for London and recommend activities based on the conditions.",
        help="Prompt for the agent to process (default: 'Please get the current weather for London and recommend activities based on the conditions')"
    )

    args = parser.parse_args()

    agent = None

    if args.agent.lower() == "gemini":
        agent = aiam.Gemini_agent_manager()
    elif args.agent.lower() == "smal":
        if args.model_id == DEF_MODEL_ID:
            model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
            args.model_id = model_id
        agent = aiam.Smal_agent_manager()
    else:
        print(f"❌ Unsupported agent family: {args.agent}")
        exit(1)

    agent.configure(model_id=args.model_id, provider=args.provider)

    response = main(agent, args.prompt)
    if response:
        print(response)
    else:
        print("No response received.")
