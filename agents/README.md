# agents

Home of the `react_agent.py`


Multi-Backend ReAct Agent with Network Instrumentation
OpenAI • Gemini • DeepSeek • Ollama (Llama, Qwen, Mistral) • Weather Tool • SerpAPI • PCAP Capture (Scapy)

This code provides a fully instrumented ReAct-style Tool-Using Agent supporting multiple LLM backends.

## LLMs

- ReAct reasoning (Thought → Action → Observation → Answer)
- Built-in tools: weather_now (Open-Meteo, no API key required)-
- Optional: Google Search / DDG / Wikipedia
✔️ Unified backend abstraction
✔️ PCAP network capture (sniffing all agent traffic) using Scapy
✔️ Automatic logs (events.jsonl, summary.jsonl)
✔️ Works on Python 3.10–3.13

## Running

You can run simply as
```
python react_agent.py --backend=gemini --temperature=0.5 --prompt="What is the weather in my place?"
```
Try
```
python react_agent.py --help
```
for the list of features.

### Agents

There are two agents to choose from:
- Weather agent, which retrieves the current weather information from the internet.
- No-tools agent, which uses only the base knownledge in the model.
- Research assistant agent.

The research assistant agent requires access to SCOPUS database.  SCOPUS API requres a key, and is limited by IP address to the subscribed institutions. To check that you have access, try:

```
curl -X GET --header 'Accept: application/json' 'https://api.elsevier.com/content/abstract/doi/10.1016/S0014-5793(01)03313-0?apiKey=9e73d803d9c7f...0e4eeafd06'
curl -X GET --header 'Accept: text/xml' 'https://api.elsevier.com/content/search/scopus?query=PUBYEAR+%3E+2018+AND+PUBYEAR+%3C+2020+AND%28TITLE%28heart+attack%29%29&view=complete&apiKey=9e73d803d...eeafd06'
```

### Docker

The agent will pick up all network trafic at the host, for as long as scapy is recording, including background e-mail activity etc.  It is therefore better to run from Docker:

```
docker build -t react-agent .
docker run --rm -e GEMINI_API_KEY=<my_AI_key> -v "$(pwd)/pcap:/app/pcap" react-agent --prompt="What is the weather forecast in London, UK?" --backend=gemini
```
### Pcap Analysis

1) Packet count burst plot
2) Transfer volume plot (KB)

To plot those plots you pass the pcap file and the client ip address

For example
```
python pcap_analysis.py ./pcap/react/openai-gpt-4o-mini-20251125-123406.pcap --client-ip=10.47.9.55

```
The client IP is used to make difference between the incoming and outgoing traffic, and the default is `172.17.0.2`, commonly used by Docker.

## Features

1. Multi-Model Backend Support

Switch between LLM providers with a simple flag:

--backend openai     # OpenAI ChatCompletions API
--backend gemini     # Google Generative Language API
--backend deepseek   # DeepSeek OpenAI-compatible API
--backend ollama      # HuggingFace Inference API

2. Create and activate environment
python3 -m venv react_env
source react_env/bin/activate

3. Install dependencies
pip install -r requirements.txt

Note
You need scapy installed for PCAP capture:
pip install scapy

4.  Environment Variables

Create a .env file:

# OpenAI
OPENAI_API_KEY=...

# Google Gemini
GEMINI_API_KEY=...

# DeepSeek (OpenAI-compatible API)
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# HuggingFace
HF_API_KEY=...

# Optional: SerpAPI for Google Search Tool
SERPAPI_API_KEY=...

Usage
1. OpenAI Example
python react_agent.py \
  "Get the current weather for London and recommend activities based on the conditions." \
  --backend openai \
  --model gpt-4o-mini \
  --serpapi

2. Gemini Example
python react_agent.py \
  "Explain the latest weather in Oslo and suggest activities." \
  --backend gemini \
  --model gemini-2.0-flash

3. DeepSeek
python react_agent.py \
  "Get NY weather and recommend activities." \
  --backend deepseek \
  --model deepseek-chat

4. LLaMA (HuggingFace)
python react_agent.py \
  "Weather in Paris and activities?" \
  --backend llama \
  --model meta-llama/Meta-Llama-3-8B-Instruct

   

pip install scapy
