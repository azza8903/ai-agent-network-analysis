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

