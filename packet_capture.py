from collections import defaultdict
import numpy as np
import pandas as pd
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

# Danh sách các thuộc tính cần tính toán
columns = [
    ' Destination Port', ' Init_Win_bytes_backward', ' Average Packet Size',
    ' Bwd Packet Length Std', 'Init_Win_bytes_forward', 'Flow Bytes/s',
    ' PSH Flag Count', 'Fwd IAT Total', 'Bwd IAT Total', ' min_seg_size_forward',
    'Total Length of Fwd Packets', ' Flow Duration', ' Packet Length Mean',
    ' Avg Bwd Segment Size', ' Bwd Packet Length Mean', ' Subflow Fwd Bytes',
    ' Fwd Packet Length Max', ' Fwd IAT Max', ' Flow IAT Max', ' ACK Flag Count',
    ' Bwd Packet Length Min', ' Max Packet Length', ' Fwd Packet Length Mean',
    ' Fwd IAT Std', ' Bwd IAT Min', ' Bwd Header Length', ' Total Backward Packets',
    'Bwd Packet Length Max', ' Packet Length Std', ' Total Fwd Packets',
    ' Subflow Bwd Packets', ' Min Packet Length', ' Subflow Bwd Bytes',
    ' Packet Length Variance', ' Fwd IAT Mean', ' act_data_pkt_fwd',
    ' URG Flag Count', 'Fwd PSH Flags', ' Flow IAT Std', ' Fwd Header Length.1',
    ' SYN Flag Count', ' Bwd Packets/s'
]

def capture_packets(interface='Wi-Fi', duration=60):
    packets = sniff(iface=interface, timeout=duration, prn=lambda x: x.summary())
    return packets

def extract_flow_key(packet):
    src = packet[IP].src
    dst = packet[IP].dst
    sport = packet[TCP].sport if TCP in packet else packet[UDP].sport
    dport = packet[TCP].dport if TCP in packet else packet[UDP].dport
    return (src, dst, sport, dport)

def calculate_packet_lengths(packets):
    return [len(pkt) for pkt in packets]

def calculate_iats(packets):
    return [packets[i].time - packets[i - 1].time for i in range(1, len(packets))]

def process_packets(packets):
    flows = defaultdict(list)

    for packet in packets:
        if IP in packet and (TCP in packet or UDP in packet):
            flow_key = extract_flow_key(packet)
            flows[flow_key].append(packet)

    data = []

    for flow_key, flow_packets in flows.items():
        if not flow_packets:
            continue

        try:
            packet_lengths = calculate_packet_lengths(flow_packets)
            iats = calculate_iats(flow_packets)
            flow_duration = flow_packets[-1].time - flow_packets[0].time if flow_packets else 0

            packet_data = {
                ' Destination Port': flow_packets[0][TCP].dport if TCP in flow_packets[0] else (
                    flow_packets[0][UDP].dport if UDP in flow_packets[0] else 0),
                ' Init_Win_bytes_backward': np.mean([pkt[TCP].window for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'window')]) if flow_packets else 0,
                ' Average Packet Size': np.mean(packet_lengths),
                ' Bwd Packet Length Std': np.std(packet_lengths),
                'Init_Win_bytes_forward': np.mean([pkt[TCP].window for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'window')]) if flow_packets else 0,
                'Flow Bytes/s': sum(packet_lengths) / flow_duration if flow_duration > 0 else 0,
                ' PSH Flag Count': sum([pkt[TCP].psh for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'psh')]),
                'Fwd IAT Total': sum(iats),
                'Bwd IAT Total': sum(iats),
                ' min_seg_size_forward': min(packet_lengths) if packet_lengths else 0,
                'Total Length of Fwd Packets': sum(packet_lengths),
                ' Flow Duration': flow_duration,
                ' Packet Length Mean': np.mean(packet_lengths) if packet_lengths else 0,
                ' Avg Bwd Segment Size': np.mean(packet_lengths) if packet_lengths else 0,
                ' Bwd Packet Length Mean': np.mean(packet_lengths) if packet_lengths else 0,
                ' Subflow Fwd Bytes': sum(packet_lengths),
                ' Fwd Packet Length Max': max(packet_lengths) if packet_lengths else 0,
                ' Fwd IAT Max': max(iats) if iats else 0,
                ' Flow IAT Max': max(iats) if iats else 0,
                ' ACK Flag Count': sum([pkt[TCP].ack for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'ack')]),
                ' Bwd Packet Length Min': min(packet_lengths) if packet_lengths else 0,
                ' Max Packet Length': max(packet_lengths) if packet_lengths else 0,
                ' Fwd Packet Length Mean': np.mean(packet_lengths) if packet_lengths else 0,
                ' Fwd IAT Std': np.std(iats) if iats else 0,
                ' Bwd IAT Min': min(iats) if iats else 0,
                ' Bwd Header Length': np.mean([len(pkt[TCP]) for pkt in flow_packets if TCP in pkt]),
                ' Total Backward Packets': len(flow_packets),
                'Bwd Packet Length Max': max(packet_lengths) if packet_lengths else 0,
                ' Packet Length Std': np.std(packet_lengths) if packet_lengths else 0,
                ' Total Fwd Packets': len(flow_packets),
                ' Subflow Bwd Packets': len(flow_packets),
                ' Min Packet Length': min(packet_lengths) if packet_lengths else 0,
                ' Subflow Bwd Bytes': sum(packet_lengths),
                ' Packet Length Variance': np.var(packet_lengths) if packet_lengths else 0,
                ' Fwd IAT Mean': np.mean(iats) if iats else 0,
                ' act_data_pkt_fwd': 0,
                ' URG Flag Count': sum([pkt[TCP].urg for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'urg')]),
                'Fwd PSH Flags': sum([pkt[TCP].psh for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'psh')]),
                ' Flow IAT Std': np.std(iats) if iats else 0,
                ' Fwd Header Length.1': np.mean([len(pkt[TCP]) for pkt in flow_packets if TCP in pkt]),
                ' SYN Flag Count': sum([pkt[TCP].syn for pkt in flow_packets if TCP in pkt and hasattr(pkt[TCP], 'syn')]),
                ' Bwd Packets/s': len(flow_packets) / flow_duration if flow_duration > 0 else 0
            }

            # Thay thế giá trị NaN và vô cùng bằng 0
            for key, value in packet_data.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    packet_data[key] = 0

            data.append(packet_data)
        except Exception as e:
            print(f"Error processing flow {flow_key}: {e}")

    return pd.DataFrame(data, columns=columns)

def get_packet_data():
    packets = capture_packets('Wi-Fi', 60)
    return process_packets(packets)
