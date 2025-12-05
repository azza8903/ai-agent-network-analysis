#from scapy.all import rdpcap, IP, IPv6, TCP, UDP
import pyshark
import ipaddress
import nest_asyncio
import pandas as pd
import sys
import requests


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
    # Filter for TCP and QUIC traffic on ports 80 and 443
    filtered_df = pcap_df[
        ((pcap_df['protocol'] == 'TCP') | (pcap_df['protocol'] == 'UDP') | (pcap_df['protocol'] == 'QUIC')) &
        ((pcap_df['source_port'].isin([80, 443, 11434])) | (pcap_df['destination_port'].isin([80, 443, 11434])))   
    ]
    return filtered_df

def analyze_pcap(pcap_file):
    pcap_df = pcap_to_dataframe(pcap_file)
    pcap_df = purify_traffic(pcap_df)

    total_bytes_sent = pcap_df[pcap_df['destination_port'].isin([80, 443])]['length'].sum()
    total_bytes_received = pcap_df[pcap_df['source_port'].isin([80, 443])]['length'].sum()
    latency_s = pcap_df['timestamp'].max()
    nr_streams = pcap_df['stream_index'].unique()
    
    return {
        "Bytes Sent (KB)": round(total_bytes_sent / 1024, 2),
        "Bytes Received (KB)": round(total_bytes_received / 1024, 2),
        "Streams": len(nr_streams),
        "Latency (s)": round(latency_s, 2)
    }


# def analyze_pcap(pcap_file):
#     packets = rdpcap(pcap_file)
#     total_bytes_sent = 0
#     total_bytes_received = 0
#     external_requests = set()
#     timestamps = []

#     for pkt in packets:
#         if IP in pkt and TCP in pkt:
#             ip_layer = pkt[IP]
#             tcp_layer = pkt[TCP]
#             packet_size = len(pkt)

#             timestamps.append(pkt.time)

#             if tcp_layer.dport in [80, 443]:
#                 total_bytes_sent += packet_size
#                 external_requests.add(ip_layer.dst)
#             elif tcp_layer.sport in [80, 443]:
#                 total_bytes_received += packet_size

#     latency_ms = 0
#     if timestamps:
#         latency_ms = (max(timestamps) - min(timestamps)) * 1000

#     return {
#         "Bytes Sent (KB)": round(total_bytes_sent / 1024, 2),
#         "Bytes Received (KB)": round(total_bytes_received / 1024, 2),
#         "External Requests": len(external_requests),
#         "Latency (ms)": round(latency_ms, 2)
#     }

def analyze_multiple_agents(pcap_files):
    results = []
    for agent_name, filepath in pcap_files.items():
        stats = analyze_pcap(filepath)
        stats["Agent"] = agent_name
        results.append(stats)
    return pd.DataFrame(results)

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