from scapy.all import sniff
import pandas as pd
import numpy as np
from collections import defaultdict

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
    packets = sniff(iface=interface, timeout=duration)
    return packets

def process_packets(packets):
    flows = defaultdict(list)

    for packet in packets:
        if 'IP' in packet and 'TCP' in packet:
            src = packet['IP'].src
            dst = packet['IP'].dst
            sport = packet['TCP'].sport
            dport = packet['TCP'].dport
            flow_key = (src, dst, sport, dport)

            flows[flow_key].append(packet)

    data = []

    for flow_key, flow_packets in flows.items():
        if not flow_packets:
            continue

        try:
            packet_data = {}
            packet_data[' Destination Port'] = flow_packets[0]['TCP'].dport if 'TCP' in flow_packets[0] else 0
            packet_data[' Init_Win_bytes_backward'] = np.mean([pkt['TCP'].window for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'window')])
            packet_data[' Average Packet Size'] = np.mean([len(pkt) for pkt in flow_packets])
            packet_data[' Bwd Packet Length Std'] = np.std([len(pkt) for pkt in flow_packets])
            packet_data['Init_Win_bytes_forward'] = np.mean([pkt['TCP'].window for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'window')])
            packet_data['Flow Bytes/s'] = sum([len(pkt) for pkt in flow_packets]) / (flow_packets[-1].time - flow_packets[0].time)
            packet_data[' PSH Flag Count'] = sum([pkt['TCP'].psh for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'psh')])
            packet_data['Fwd IAT Total'] = sum([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data['Bwd IAT Total'] = sum([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' min_seg_size_forward'] = min([len(pkt) for pkt in flow_packets])
            packet_data['Total Length of Fwd Packets'] = sum([len(pkt) for pkt in flow_packets])
            packet_data[' Flow Duration'] = flow_packets[-1].time - flow_packets[0].time
            packet_data[' Packet Length Mean'] = np.mean([len(pkt) for pkt in flow_packets])
            packet_data[' Avg Bwd Segment Size'] = np.mean([len(pkt) for pkt in flow_packets])
            packet_data[' Bwd Packet Length Mean'] = np.mean([len(pkt) for pkt in flow_packets])
            packet_data[' Subflow Fwd Bytes'] = sum([len(pkt) for pkt in flow_packets])
            packet_data[' Fwd Packet Length Max'] = max([len(pkt) for pkt in flow_packets])
            packet_data[' Fwd IAT Max'] = max([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' Flow IAT Max'] = max([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' ACK Flag Count'] = sum([pkt['TCP'].ack for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'ack')])
            packet_data[' Bwd Packet Length Min'] = min([len(pkt) for pkt in flow_packets])
            packet_data[' Max Packet Length'] = max([len(pkt) for pkt in flow_packets])
            packet_data[' Fwd Packet Length Mean'] = np.mean([len(pkt) for pkt in flow_packets])
            packet_data[' Fwd IAT Std'] = np.std([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' Bwd IAT Min'] = min([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' Bwd Header Length'] = np.mean([len(pkt['TCP']) for pkt in flow_packets if 'TCP' in pkt])
            packet_data[' Total Backward Packets'] = len(flow_packets)
            packet_data['Bwd Packet Length Max'] = max([len(pkt) for pkt in flow_packets])
            packet_data[' Packet Length Std'] = np.std([len(pkt) for pkt in flow_packets])
            packet_data[' Total Fwd Packets'] = len(flow_packets)
            packet_data[' Subflow Bwd Packets'] = len(flow_packets)
            packet_data[' Min Packet Length'] = min([len(pkt) for pkt in flow_packets])
            packet_data[' Subflow Bwd Bytes'] = sum([len(pkt) for pkt in flow_packets])
            packet_data[' Packet Length Variance'] = np.var([len(pkt) for pkt in flow_packets])
            packet_data[' Fwd IAT Mean'] = np.mean([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' act_data_pkt_fwd'] = 0
            packet_data[' URG Flag Count'] = sum([pkt['TCP'].urg for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'urg')])
            packet_data['Fwd PSH Flags'] = sum([pkt['TCP'].psh for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'psh')])
            packet_data[' Flow IAT Std'] = np.std([flow_packets[i].time - flow_packets[i - 1].time for i in range(1, len(flow_packets))])
            packet_data[' Fwd Header Length.1'] = np.mean([len(pkt['TCP']) for pkt in flow_packets if 'TCP' in pkt])
            packet_data[' SYN Flag Count'] = sum([pkt['TCP'].syn for pkt in flow_packets if 'TCP' in pkt and hasattr(pkt['TCP'], 'syn')])
            packet_data[' Bwd Packets/s'] = len(flow_packets) / (flow_packets[-1].time - flow_packets[0].time)

            data.append(packet_data)
        except Exception as e:
            print(f"Error processing flow {flow_key}: {e}")

    return pd.DataFrame(data, columns=columns)

# Ví dụ về cách sử dụng (sẽ được sử dụng trong main.py)
def get_packet_data(interface='Wi-Fi', duration=60):
    packets = capture_packets(interface, duration)
    return process_packets(packets)
