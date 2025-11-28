# experiments

Directory for systematic experiment execution.

## Usage

Create scripts called `run{ID_or_name}.py`. They should run Docker container react-agent.

⚠️ For how to build react-agent, see `../agents/README` ⚠️

## Contents
- `run1.py` — Sample (first) experiment running `react-agent` with different backends.
- `pcap/` — PCAP files produced by the experiments.  Subdirectories `pcap/run{ID_or_name}/` will be created automatically.
- `output/` — outputs produced by container runs. Subdirectories `output/run{ID_or_name}/` will be created automatically.

