react_agent.py

Multi-Backend ReAct Agent with Network Instrumentation
OpenAI • Gemini • DeepSeek • LLaMA (Hugging Face) • Weather Tool • SerpAPI • PCAP Capture (Scapy)

This code provides a fully instrumented ReAct-style Tool-Using Agent supporting multiple LLM backends:

- OpenAI (GPT-4o, GPT-4o-mini, etc.)

- Google Gemini (v1.5, v2.0, v2.5)

- DeepSeek via OpenAI API compatibility

= HuggingFace Inference API (Meta-LLaMA, Mistral, Qwen, etc.)

It includes:

- ReAct reasoning (Thought → Action → Observation → Answer)
- Built-in tools: weather_now (Open-Meteo, no API key required)-
- Optional: Google Search / DDG / Wikipedia
✔️ Unified backend abstraction
✔️ PCAP network capture (sniffing all agent traffic) using Scapy
✔️ Automatic logs (events.jsonl, summary.jsonl)
✔️ Works on Python 3.10–3.13

1. Multi-Model Backend Support

Switch between LLM providers with a simple flag:

--backend openai     # OpenAI ChatCompletions API
--backend gemini     # Google Generative Language API
--backend deepseek   # DeepSeek OpenAI-compatible API
--backend llama      # HuggingFace Inference API

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
