# 🧠 Custom Agents for Network Analysis

This folder contains custom agents and configuration files built on top of the [`smolagents`](https://github.com/huggingface/smolagents) framework. These agents are used in network-level behavior experiments.

## 📂 Project Structure

All code is placed under:

smolagents/examples/open_deep_research/


## ▶️ Running the Agents

To run the agents:

1. Make sure you've installed the `smolagents` project correctly using its installation instructions.
2. Use a Python virtual environment for isolation (recommended).
3. Navigate to the `open_deep_research/` folder and run your agent code (e.g., `open_source_models.py` or similar).

### Example

To run with the default model (Qwen/Qwen2.5-Coder-32B-Instruct):

```bash
cd smolagents/examples/open_deep_research/
python open_source_models.py
```
To specify a model:

```bash

python open_source_models.py --model-id meta-llama/Llama-3-8B-Instruct

```

Analyze the Traffic (Optional)
If scapy is installed and you're running with root/administrator privileges, the script will:

- Start a packet sniffer
- Save results in agent_traffic.pcap

You can open this file with Wireshark or analyze it with Python.


## 🐍 Virtual Environment Setup
If you're using a virtual environment (recommended), here’s how to set it up:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

