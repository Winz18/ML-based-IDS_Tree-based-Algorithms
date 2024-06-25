import pandas as pd
from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP

# Khởi tạo danh sách các thuộc tính
attributes = [
    'destination_port', 'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
    'total_length_fwd_packets', 'total_length_bwd_packets', 'fwd_packet_length_max',
    'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std',
    'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean',
    'bwd_packet_length_std', 'flow_bytes_per_s', 'flow_packets_per_s', 'flow_iat_mean',
    'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean',
    'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean',
    'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags',
    'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_length', 'bwd_header_length',
    'fwd_packets_per_s', 'bwd_packets_per_s', 'min_packet_length', 'max_packet_length',
    'packet_length_mean', 'packet_length_std', 'packet_length_variance', 'fin_flag_count',
    'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
    'cwe_flag_count', 'ece_flag_count', 'down_up_ratio', 'average_packet_size',
    'fwd_segment_size_avg', 'bwd_segment_size_avg', 'fwd_bytes_per_bulk_avg',
    'fwd_packet_per_bulk_avg', 'fwd_bulk_rate_avg', 'bwd_bytes_per_bulk_avg',
    'bwd_packet_per_bulk_avg', 'bwd_bulk_rate_avg', 'subflow_fwd_packets', 'subflow_fwd_bytes',
    'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_win_bytes_forward',
    'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'active_mean',
    'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min'
]

# Tạo DataFrame để lưu trữ dữ liệu
df = pd.DataFrame(columns=attributes)

# Lưu trữ thông tin các luồng
flows = defaultdict(lambda: {
    'total_fwd_packets': 0,
    'total_bwd_packets': 0,
    'total_length_fwd_packets': 0,
    'total_length_bwd_packets': 0,
    'fwd_packet_lengths': [],
    'bwd_packet_lengths': [],
    'flow_start_time': None,
    'flow_end_time': None,
    'fwd_iat': [],
    'bwd_iat': [],
    'fwd_psh_flags': 0,
    'bwd_psh_flags': 0,
    'fwd_urg_flags': 0,
    'bwd_urg_flags': 0,
    'fwd_header_length': 0,
    'bwd_header_length': 0,
    'fin_flag_count': 0,
    'syn_flag_count': 0,
    'rst_flag_count': 0,
    'psh_flag_count': 0,
    'ack_flag_count': 0,
    'urg_flag_count': 0,
    'cwe_flag_count': 0,
    'ece_flag_count': 0
})


def calculate_statistics(packet_lengths):
    if not packet_lengths:
        return 0, 0, 0, 0
    return (
        max(packet_lengths),
        min(packet_lengths),
        sum(packet_lengths) / len(packet_lengths),
        pd.Series(packet_lengths).std()
    )


def process_packet(packet):
    global df
    if IP in packet:
        destination_port = packet[IP].dport
        flow = flows[destination_port]

        current_time = packet.time

        if flow['flow_start_time'] is None:
            flow['flow_start_time'] = current_time

        flow['flow_end_time'] = current_time
        flow_duration = flow['flow_end_time'] - flow['flow_start_time']

        if TCP in packet or UDP in packet:
            destination_port = packet.dport
        else:
            destination_port = None

        packet_length = len(packet)

        if packet[IP].src == packet[IP].src:
            flow['total_fwd_packets'] += 1
            flow['total_length_fwd_packets'] += packet_length
            flow['fwd_packet_lengths'].append(packet_length)
            if flow['total_fwd_packets'] > 1:
                flow['fwd_iat'].append(current_time - flow['flow_end_time'])
        else:
            flow['total_bwd_packets'] += 1
            flow['total_length_bwd_packets'] += packet_length
            flow['bwd_packet_lengths'].append(packet_length)
            if flow['total_bwd_packets'] > 1:
                flow['bwd_iat'].append(current_time - flow['flow_end_time'])

        if TCP in packet:
            flags = packet[TCP].flags
            if flags & 0x01: flow['fin_flag_count'] += 1
            if flags & 0x02: flow['syn_flag_count'] += 1
            if flags & 0x04: flow['rst_flag_count'] += 1
            if flags & 0x08: flow['psh_flag_count'] += 1
            if flags & 0x10: flow['ack_flag_count'] += 1
            if flags & 0x20: flow['urg_flag_count'] += 1
            if flags & 0x40: flow['ece_flag_count'] += 1
            if flags & 0x80: flow['cwe_flag_count'] += 1

        packet_data = {
            'destination_port': destination_port,
            'flow_duration': flow_duration,
            'total_fwd_packets': flow['total_fwd_packets'],
            'total_bwd_packets': flow['total_bwd_packets'],
            'total_length_fwd_packets': flow['total_length_fwd_packets'],
            'total_length_bwd_packets': flow['total_length_bwd_packets'],
            'fwd_packet_length_max': max(flow['fwd_packet_lengths'], default=0),
            'fwd_packet_length_min': min(flow['fwd_packet_lengths'], default=0),
            'fwd_packet_length_mean': sum(flow['fwd_packet_lengths']) / len(flow['fwd_packet_lengths']) if flow[
                'fwd_packet_lengths'] else 0,
            'fwd_packet_length_std': pd.Series(flow['fwd_packet_lengths']).std() if flow['fwd_packet_lengths'] else 0,
            'bwd_packet_length_max': max(flow['bwd_packet_lengths'], default=0),
            'bwd_packet_length_min': min(flow['bwd_packet_lengths'], default=0),
            'bwd_packet_length_mean': sum(flow['bwd_packet_lengths']) / len(flow['bwd_packet_lengths']) if flow[
                'bwd_packet_lengths'] else 0,
            'bwd_packet_length_std': pd.Series(flow['bwd_packet_lengths']).std() if flow['bwd_packet_lengths'] else 0,
            'flow_bytes_per_s': (flow['total_length_fwd_packets'] + flow[
                'total_length_bwd_packets']) / flow_duration if flow_duration > 0 else 0,
            'flow_packets_per_s': (flow['total_fwd_packets'] + flow[
                'total_bwd_packets']) / flow_duration if flow_duration > 0 else 0,
            'flow_iat_mean': sum(flow['fwd_iat'] + flow['bwd_iat']) / (len(flow['fwd_iat']) + len(flow['bwd_iat'])) if (
                    flow['fwd_iat'] + flow['bwd_iat']) else 0,
            'flow_iat_std': pd.Series(flow['fwd_iat'] + flow['bwd_iat']).std() if (
                    flow['fwd_iat'] + flow['bwd_iat']) else 0,
            'flow_iat_max': max(flow['fwd_iat'] + flow['bwd_iat'], default=0),
            'flow_iat_min': min(flow['fwd_iat'] + flow['bwd_iat'], default=0),
            'fwd_iat_total': sum(flow['fwd_iat']),
            'fwd_iat_mean': sum(flow['fwd_iat']) / len(flow['fwd_iat']) if flow['fwd_iat'] else 0,
            'fwd_iat_std': pd.Series(flow['fwd_iat']).std() if flow['fwd_iat'] else 0,
            'fwd_iat_max': max(flow['fwd_iat'], default=0),
            'fwd_iat_min': min(flow['fwd_iat'], default=0),
            'bwd_iat_total': sum(flow['bwd_iat']),
            'bwd_iat_mean': sum(flow['bwd_iat']) / len(flow['bwd_iat']) if flow['bwd_iat'] else 0,
            'bwd_iat_std': pd.Series(flow['bwd_iat']).std() if flow['bwd_iat'] else 0,
            'bwd_iat_max': max(flow['bwd_iat'], default=0),
            'bwd_iat_min': min(flow['bwd_iat'], default=0),
            'fwd_psh_flags': flow['fwd_psh_flags'],
            'bwd_psh_flags': flow['bwd_psh_flags'],
            'fwd_urg_flags': flow['fwd_urg_flags'],
            'bwd_urg_flags': flow['bwd_urg_flags'],
            'fwd_header_length': flow['fwd_header_length'],
            'bwd_header_length': flow['bwd_header_length'],
            'fwd_packets_per_s': flow['total_fwd_packets'] / flow_duration if flow_duration > 0 else 0,
            'bwd_packets_per_s': flow['total_bwd_packets'] / flow_duration if flow_duration > 0 else 0,
            'min_packet_length': min(flow['fwd_packet_lengths'] + flow['bwd_packet_lengths'], default=0),
            'max_packet_length': max(flow['fwd_packet_lengths'] + flow['bwd_packet_lengths'], default=0),
            'packet_length_mean': sum(flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']) / (
                    len(flow['fwd_packet_lengths']) + len(flow['bwd_packet_lengths'])) if (
                    flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']) else 0,
            'packet_length_std': pd.Series(flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']).std() if (
                    flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']) else 0,
            'packet_length_variance': pd.Series(flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']).var() if (
                    flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']) else 0,
            'fin_flag_count': flow['fin_flag_count'],
            'syn_flag_count': flow['syn_flag_count'],
            'rst_flag_count': flow['rst_flag_count'],
            'psh_flag_count': flow['psh_flag_count'],
            'ack_flag_count': flow['ack_flag_count'],
            'urg_flag_count': flow['urg_flag_count'],
            'cwe_flag_count': flow['cwe_flag_count'],
            'ece_flag_count': flow['ece_flag_count'],
            'down_up_ratio': flow['total_length_bwd_packets'] / flow['total_length_fwd_packets'] if flow[
                                                                                                        'total_length_fwd_packets'] > 0 else 0,
            'average_packet_size': sum(flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']) / (
                    len(flow['fwd_packet_lengths']) + len(flow['bwd_packet_lengths'])) if (
                    flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']) else 0,
            'fwd_segment_size_avg': sum(flow['fwd_packet_lengths']) / flow['total_fwd_packets'] if flow[
                                                                                                       'total_fwd_packets'] > 0 else 0,
            'bwd_segment_size_avg': sum(flow['bwd_packet_lengths']) / flow['total_bwd_packets'] if flow[
                                                                                                       'total_bwd_packets'] > 0 else 0,
            'fwd_bytes_per_bulk_avg': 0,  # Placeholder
            'fwd_packet_per_bulk_avg': 0,  # Placeholder
            'fwd_bulk_rate_avg': 0,  # Placeholder
            'bwd_bytes_per_bulk_avg': 0,  # Placeholder
            'bwd_packet_per_bulk_avg': 0,  # Placeholder
            'bwd_bulk_rate_avg': 0,  # Placeholder
            'subflow_fwd_packets': flow['total_fwd_packets'],
            'subflow_fwd_bytes': flow['total_length_fwd_packets'],
            'subflow_bwd_packets': flow['total_bwd_packets'],
            'subflow_bwd_bytes': flow['total_length_bwd_packets'],
            'init_win_bytes_forward': 0,  # Placeholder
            'init_win_bytes_backward': 0,  # Placeholder
            'act_data_pkt_fwd': 0,  # Placeholder
            'min_seg_size_forward': 0,  # Placeholder
            'active_mean': 0,  # Placeholder
            'active_std': 0,  # Placeholder
            'active_max': 0,  # Placeholder
            'active_min': 0,  # Placeholder
            'idle_mean': 0,  # Placeholder
            'idle_std': 0,  # Placeholder
            'idle_max': 0,  # Placeholder
            'idle_min': 0  # Placeholder
        }

        df = df._append(packet_data, ignore_index=True)


# Đặt thời gian thu thập (tính bằng giây)
capture_duration = 10

# Lưu thời gian bắt đầu
start_time = time.time()

# Sử dụng vòng lặp while với điều kiện thời gian
while time.time() - start_time <= capture_duration:
    sniff(iface='Wi-Fi', prn=process_packet, count=1)

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(df)
print(df.head())

# Lưu dữ liệu vào tệp CSV
time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
df.to_csv(f'result_{time_stamp}.csv', index=False)

