from typing import List, Tuple, Optional, Dict
from scapy.all import rdpcap, IP, IPv6
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

PacketTuple = Tuple[float, int, int]
# (t_rel, direction, size)


def debug_client_ip_matches(pcap_path: str, client_ip: str, max_packets: Optional[int] = 1000):
    pkts = rdpcap(pcap_path)
    total = 0
    matches = 0
    for i, pkt in enumerate(pkts):
        if max_packets and i >= max_packets:
            break

        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
        elif IPv6 in pkt:
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst
        else:
            continue

        total += 1
        if src_ip == client_ip or dst_ip == client_ip:
            matches += 1

    print(f"[DEBUG] In first {total} IP packets, {matches} involve client_ip={client_ip!r}")


def pcap_to_trace_scapy(
    pcap_path: str,
    client_ip: str,
    max_packets: Optional[int] = None,
) -> List[PacketTuple]:

    print(f"Loading pcap: {pcap_path}")
    pkts = rdpcap(pcap_path)
    print(f"Total packets loaded: {len(pkts)}")
    print(f"Using client_ip = {client_ip}")

    trace = []
    first_ts = None
    matched = 0
    processed = 0

    for i, pkt in enumerate(pkts):
        if max_packets and processed >= max_packets:
            break
        processed += 1

        # Only IPv4/IPv6 packets
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
        elif IPv6 in pkt:
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst
        else:
            continue

        # Direction is determined by the client IP
        if src_ip == client_ip:
            direction = +1
        elif dst_ip == client_ip:
            direction = -1
        else:
            continue  # not our flow

        matched += 1

        ts = float(pkt.time)
        size_bytes = len(pkt)

        if first_ts is None:
            first_ts = ts

        t_rel = ts - first_ts

        trace.append((t_rel, direction, size_bytes))

    print(f"Processed packets: {processed}")
    print(f"Packets involving client_ip: {matched}")

    return sorted(trace, key=lambda x: x[0])


def build_mtam(
    trace: List[PacketTuple],
    window_size: float = 0.1,   # 100 ms
    num_windows: int = 600,     # 60 seconds total
    clip_to_num_windows: bool = True,
) -> np.ndarray:
    if not trace:
        print("Warning: empty trace, returning zeros.")
        return np.zeros((4, num_windows), dtype=np.float32)

    last_t = trace[-1][0]
    max_index = int(last_t // window_size)

    if not clip_to_num_windows and max_index + 1 > num_windows:
        num_windows = max_index + 1

    N_in = np.zeros(num_windows, dtype=np.float32)
    N_out = np.zeros(num_windows, dtype=np.float32)
    B_in = np.zeros(num_windows, dtype=np.float32)
    B_out = np.zeros(num_windows, dtype=np.float32)

    for t_rel, direction, size_bytes in trace:
        idx = int(t_rel // window_size)
        if idx < 0 or idx >= num_windows:
            if clip_to_num_windows:
                continue
            else:
                continue

        if direction == -1:
            N_in[idx] += 1
            B_in[idx] += size_bytes
        elif direction == +1:
            N_out[idx] += 1
            B_out[idx] += size_bytes

    mtam = np.stack([N_in, N_out, B_in, B_out], axis=0)
    return mtam


def build_counts_for_plot(
    trace,
    window_size=0.05,   # 50 ms windows = fine-grained for burst visualization
):
    if not trace:
        return None, None, None

    last_t = trace[-1][0]
    num_windows = int(last_t // window_size) + 1

    n_in  = np.zeros(num_windows, dtype=np.int32)
    n_out = np.zeros(num_windows, dtype=np.int32)

    for t_rel, direction, size in trace:
        idx = int(t_rel // window_size)
        if idx >= num_windows:
            continue
        if direction == -1:
            n_in[idx] += 1
        else:
            n_out[idx] += 1

    times = np.arange(num_windows) * window_size
    return times, n_in, n_out


def plot_packet_bursts(times, n_in, n_out, title="Packet Bursts Over Time",save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(times, n_out, label="Outgoing packets", linewidth=1.5)
    plt.plot(times, n_in, label="Incoming packets", linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Packet Count (per window)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[+] Saved figure to: {save_path}")
    plt.close()

def build_volume_for_plot(
    trace,
    window_size=0.05,   # same idea: per-window aggregation
):
    """
    Return:
        times   -> array of time window centers
        kb_in   -> incoming KB per window
        kb_out  -> outgoing KB per window
    """
    if not trace:
        return None, None, None

    last_t = trace[-1][0]
    num_windows = int(last_t // window_size) + 1

    bytes_in  = np.zeros(num_windows, dtype=np.float64)
    bytes_out = np.zeros(num_windows, dtype=np.float64)

    for t_rel, direction, size_bytes in trace:
        idx = int(t_rel // window_size)
        if idx >= num_windows:
            continue
        if direction == -1:
            bytes_in[idx] += size_bytes
        else:
            bytes_out[idx] += size_bytes

    # Convert to KB
    kb_in  = bytes_in  / 1024.0
    kb_out = bytes_out / 1024.0

    times = np.arange(num_windows) * window_size
    return times, kb_in, kb_out

def plot_transfer_volume(times, kb_in, kb_out, title="Transfer Volume Over Time (KB)", save_path=None):
    plt.figure(figsize=(12, 4))

    plt.plot(times, kb_out, label="Outgoing volume (KB)", linewidth=1.5)
    plt.plot(times, kb_in,  label="Incoming volume (KB)", linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.ylabel("KB per window")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[+] Saved volume figure to: {save_path}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM agent PCAP patterns")
    parser.add_argument("--pcap", type=str, help="Path to input pcap file")
    parser.add_argument("--client_ip", type=str, default="172.17.0.2", help="Client IP in the capture")

    args = parser.parse_args()

    #args.pcap = "experiments/pcap/run-ollama-models/ollama-mistral-20251203-101258.pcap"
    #args.pcap = "pcap/run_research_agent_100/deepseek-deepseek-chat-20251205-120944.pcap"

    debug_client_ip_matches(args.pcap, args.client_ip)

    trace = pcap_to_trace_scapy(args.pcap, args.client_ip)

    print("Extracted packets:", len(trace))
    print("Sample:", trace[:5])

    mtam = build_mtam(trace, window_size=0.1, num_windows=600)
    print("MTAM shape:", mtam.shape)
    print("N_in first 10 windows:", mtam[0, :10])
    print("N_out first 10 windows:", mtam[1, :10])

    print("Total packets (trace):", len(trace))
    print("Total packets in MTAM:",
      int(mtam[0].sum() + mtam[1].sum()))

    print("Total bytes in MTAM:",
      int(mtam[2].sum() + mtam[3].sum()))

    # Extract base name (remove directories + extension)
    base = os.path.basename(args.pcap)
    name_no_ext = os.path.splitext(base)[0] 
    # Build output file name
    out_png = f"{name_no_ext}.png"

    # 1) Packet count burst plot
    times, n_in, n_out = build_counts_for_plot(trace, window_size=0.01)
    if times is not None:
        out_png_counts = f"{name_no_ext}_counts.png"
        plot_packet_bursts(
            times, n_in, n_out,
            title=f"{name_no_ext} – Packet Counts",
            save_path=out_png_counts,
        )
    # 2) Transfer volume plot (KB)
    times_v, kb_in, kb_out = build_volume_for_plot(trace, window_size=0.01)
    if times_v is not None:
        out_png_volume = f"{name_no_ext}_volume.png"
        plot_transfer_volume(
            times_v, kb_in, kb_out,
            title=f"{name_no_ext} – Transfer Volume (KB)",
            save_path=out_png_volume,
        )

if __name__ == "__main__":
    main()
