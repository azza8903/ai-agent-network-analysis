# top of file
from dotenv import dotenv_values
import os
import shlex
import subprocess
from pathlib import Path
from datetime import datetime
import json

env_map = dotenv_values()  # returns a dict of key->value
# number of times to run each container
RUNS = 100

# fixed prompt and container image
PROMPTS_PATH = Path(__file__).parent / "prompts.json"

if PROMPTS_PATH.exists():
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        PROMPTS = data
    elif isinstance(data, dict):
        PROMPTS = data.get("prompts") or [v for v in data.values() if isinstance(v, str)]
    else:
        PROMPTS = []
else:
    # fallback if the file is empty / invalid
    PROMPTS = ["What is the weather forecast in London, UK?"]

PROMPT_ID = 11
AGENT_TYPE = PROMPTS[PROMPT_ID][0]
PROMPT = PROMPTS[PROMPT_ID][1] 
IMAGE = "react-agent"
PCAP_HOST_PATH = str(Path.cwd() / "pcap")
OUTPUT_HOST_PATH = str(Path.cwd() / "output")

script_name = Path(__file__).stem

pcap_path = Path(PCAP_HOST_PATH) / script_name
pcap_path.mkdir(parents=True, exist_ok=True) # PCAP path must exist or else the docker run will suffer

# backends description: env key, base flags and optional models list
backends = {
    "gemini": {"env_key": "GEMINI_API_KEY", "flags": ["--backend=gemini"]},
    "deepseek": {"env_key": "DEEPSEEK_API_KEY", "flags": ["--backend=deepseek"]},
    "openai": {"env_key": "OPENAI_API_KEY", "flags": ["--backend=openai"]},
    # "ollama": {
    #     "env_key": None,
    #     "flags": ["--backend=ollama"],
    #     "models": ["mistral", "llama3.1", "qwen3:8b"],
    # },
}

def _run_docker(backend_name: str, model_name: str, env_key: str | None = None, env_val: str | None = None, extra_flags: list | None = None):
    extra_flags = extra_flags or []

    # build base command
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v", "/etc/localtime:/etc/localtime",
        "-v", "/etc/timezone:/etc/timezone",    
        "-v",
        f"{PCAP_HOST_PATH}:/app/pcap",
        IMAGE,
        f"--agent={AGENT_TYPE}",
        f"--prompt={PROMPT}",
        *extra_flags,
    ]

    # only pass env var if both key and value are provided
    if env_key and env_val:
        cmd[3:3] = ["-e", f"{env_key}={env_val}"]  # insert after "run" and "--rm"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = Path(f"{OUTPUT_HOST_PATH}/{script_name}/{backend_name}_{model_name}_{ts}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # run once, capture stdout/stderr, write to file and raise on non-zero exit
    proc = subprocess.run(cmd, capture_output=True, text=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(shlex.quote(x) for x in cmd)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n\n--- STDERR ---\n")
            f.write(proc.stderr)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

    print("Running:", " ".join(shlex.quote(x) for x in cmd))


for _ in range(RUNS):
    print(f"===== RUN {_ + 1} =====")
    for backend_name, cfg in backends.items():
        env_key = cfg["env_key"]
        env_val = env_map.get(env_key) if env_key else None  # Local LLMs don't need a key

        # include pcap-dir for all runs
        base_flags = [*cfg["flags"], f"--pcap-dir=pcap/{script_name}"]

        if backend_name == "ollama":
            for model in cfg.get("models", []):
                flags = [*base_flags, f"--model={model}"]
                try:
                    _run_docker(backend_name, model, env_key, env_val, flags)
                except subprocess.CalledProcessError as exc:
                    print(f"docker run failed for {backend_name} model={model}: {exc}")
        else:
            flags = base_flags
            try:
                _run_docker(backend_name, "default", env_key, env_val, flags)
            except subprocess.CalledProcessError as exc:
                print(f"docker run failed for {backend_name}: {exc}")