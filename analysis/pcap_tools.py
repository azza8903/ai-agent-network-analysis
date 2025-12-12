# Top of the file: analysis/pcap_tools.py
# The current version of this library uses both Scapy and Pyshark for different functionalities.

from scapy.all import rdpcap, IP, IPv6, TCP, UDP
import pyshark
import ipaddress
import nest_asyncio
import pandas as pd
import sys
import requests
from typing import Optional, List, Tuple
import numpy as np

PacketTuple = Tuple[float, int, int]


# Apply nest_asyncio to the current event loop
nest_asyncio.apply()

def is_local_or_multicast(ip_str):
    """
    Checks if an IP address string is a private, loopback, or multicast address.
    """
    if ip_str is None:
        return False
    try:
        # The factory function automatically detects IPv4 or IPv6
        ip_addr = ipaddress.ip_address(ip_str)
        
        # Check for private, loopback, and multicast addresses
        return ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_multicast
    except ValueError:
        # Handles cases where ip_str is not a valid IP address
        return False

def filter_local_traffic(df):
    """
    Filters a pandas DataFrame to remove packets where both the source and
    destination IPs are local or multicast addresses.
    """
    # Create a boolean mask where both source and destination are local/multicast
    mask = df.apply(
        lambda row: is_local_or_multicast(row['source_ip']) and
                    is_local_or_multicast(row['destination_ip']),
        axis=1
    )
    
    # Invert the mask to keep only the packets that do not match the criteria
    filtered_df = df[~mask].copy()
    
    return filtered_df

def purify_traffic(pcap_df):
    # Filter for TCP and QUIC traffic on ports 80 and 443, as well as 11434 for Ollama
    filtered_df = pcap_df[
        ((pcap_df['protocol'] == 'TCP') | (pcap_df['protocol'] == 'UDP') | (pcap_df['protocol'] == 'QUIC')) &
        ((pcap_df['source_port'].isin([80, 443, 11434])) | (pcap_df['destination_port'].isin([80, 443, 11434])))   
    ]
    return filtered_df

def analyze_pcap(pcap_file):
    pcap_df = pcap_to_dataframe(pcap_file)
    pcap_df = purify_traffic(pcap_df)

    total_bytes_sent = pcap_df[pcap_df['destination_port'].isin([80, 443, 11434])]['length'].sum()
    total_bytes_received = pcap_df[pcap_df['source_port'].isin([80, 443, 11434])]['length'].sum()
    latency_s = pcap_df['timestamp'].max()
    nr_streams = pcap_df['stream_index'].unique()
    
    return {
        "Bytes Sent (KB)": round(total_bytes_sent / 1024, 2),
        "Bytes Received (KB)": round(total_bytes_received / 1024, 2),
        "Streams": len(nr_streams),
        "Latency (s)": round(latency_s, 2)
    }

def pcap_to_trace_scapy(
    pcap_path: str,
    client_ip: str,
    max_packets: Optional[int] = None,
) -> List[PacketTuple]:

    #print(f"Loading pcap: {pcap_path}")
    pkts = rdpcap(pcap_path)
    #print(f"Total packets loaded: {len(pkts)}")
    #print(f"Using client_ip = {client_ip}")

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

    #print(f"Processed packets: {processed}")
    #print(f"Packets involving client_ip: {matched}")

    if matched / len(pkts) < 0.8:
        print(f"*** Warning: less than 80% of packets involve client_ip={client_ip!r} ({matched} out of {len(pkts)})")

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

    if max_index + 1 > num_windows:
        print(f"*** Warning: trace length exceeds num_windows ({max_index + 1} > {num_windows})")

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

def pcap_to_dataframe(pcap_file):
    """
    Reads a pcap file and converts it into a pandas DataFrame.
    
    Args:
        pcap_file (str): The path to the pcap file.
        
    Returns:
        pandas.DataFrame: A DataFrame containing packet information.

        The dataframe includes the protocol used (TCP, UDP, QUIC), and stream index from pyshark, in addition to the ususal fields.
    """
    try:
        # Create a pyshark file capture object
        cap = pyshark.FileCapture(pcap_file)
        
        # Prepare a list to store packet data
        packets_data = []
        first_packet_timestamp = None
        
        # Iterate over each packet in the capture
        for packet in cap:
            timestamp = packet.sniff_time
            if first_packet_timestamp is None:
                first_packet_timestamp = timestamp
            
            timestamp = (timestamp - first_packet_timestamp).total_seconds()
            protocol = None
            src_ip = None
            dst_ip = None
            src_port = None
            dst_port = None
            stream_id= None

            # Check for the highest-level protocol
            # Pyshark automatically detects QUIC due to Wireshark's dissector
            if 'QUIC' in packet:
                protocol = 'QUIC'
                if 'udp' in packet:
                    src_port = packet.udp.srcport
                    dst_port = packet.udp.dstport
                if hasattr(packet.udp, 'stream'):
                    # QUIC is typically encrypted, and tshark cannot access the flow ID directly. 
                    #  To identify QUIC streams, we can use the UDP stream index with an offset.
                    stream_id = 10000 + int(packet.udp.stream)
            elif 'TCP' in packet:
                protocol = 'TCP'
                src_port = packet.tcp.srcport
                dst_port = packet.tcp.dstport
                if hasattr(packet.tcp, 'stream'):
                    stream_id =  int(packet.tcp.stream)  
            elif 'UDP' in packet:
                # If it's a generic UDP packet, but not identified as QUIC
                protocol = 'UDP'
                src_port = packet.udp.srcport
                dst_port = packet.udp.dstport
                if hasattr(packet.udp, 'stream'):
                    stream_id = int(packet.udp.stream)

            # Handle IPv4 and IPv6 layers
            if 'IP' in packet:
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
            elif 'IPV6' in packet:
                src_ip = packet.ipv6.src
                dst_ip = packet.ipv6.dst
            
            if protocol:
                # Append the extracted data to our list
                packets_data.append({
                    'timestamp': timestamp,
                    'source_ip': src_ip,
                    'destination_ip': dst_ip,
                    'protocol': protocol,
                    'length': packet.length,
                    'source_port': src_port,
                    'destination_port': dst_port,
                    'stream_index': stream_id
                })
        
        # Close the capture file
        cap.close()

        # Create the DataFrame
        df = pd.DataFrame(packets_data)
        #df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        df['length'] = pd.to_numeric(df['length']).astype('Int64')
        df['source_port'] = pd.to_numeric(df['source_port'], errors='coerce').astype('Int64')
        df['destination_port'] = pd.to_numeric(df['destination_port'], errors='coerce').astype('Int64')
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Simple IP lookup with caching
ip_cache = {}
def lookup_ip(ip):
    if ip in ip_cache:
        return ip_cache[ip]

    try:
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        data = response.json()
        result = (data.get("as", ""), data.get("countryCode", ""))
        ip_cache[ip] = result
        return result
    except Exception as e:
        print ("Lookup failed at " + ip, file=sys.stderr)
        sys.stderr.write("Lookup failed at " + ip)
        return "", ""  # fallback if the request fails