import pandas as pd
import statistics
from collections import defaultdict
import joblib
import ipaddress

def ip_to_numeric(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return None  # Per manejar IPs mal formades

def prepare_data(real_data):

    real_data['SourceIP'] = real_data['SourceIP'].apply(ip_to_numeric)
    real_data['DestinationIP'] = real_data['DestinationIP'].apply(ip_to_numeric)

    shuffled_features = real_data.apply(pd.to_numeric, errors='coerce')

    # If any NaN, convert it to 0s
    shuffled_features = shuffled_features.fillna(0)
    return shuffled_features

def process_packets(packet_queue):
    # Dictionary to track flow data
    flows = defaultdict(lambda: {
        'bytes_sent': 0,
        'bytes_received': 0,
        'timestamps': [],
        'packet_lengths': [],
        'response_times': []
    })
    rf_model = joblib.load('./rf.joblib')
        
    while True:
        packet = packet_queue.get()
        if packet is None:
            break

        try:
            # Extract source and destination IP addresses and ports
            src_ip = packet.ip.src if hasattr(packet, 'ip') else "Unknown"
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else "Unknown"
            src_port = packet.tcp.srcport if hasattr(packet, 'tcp') else "Unknown"
            dst_port = packet.tcp.dstport if hasattr(packet, 'tcp') else "Unknown"

            # Extract timestamp and packet length
            timestamp = packet.sniff_time
            packet_length = int(packet.length)

            # Define flow key
            flow_key = (src_ip, dst_ip, src_port, dst_port)

            # Update flow statistics
            flow = flows[flow_key]
            flow['bytes_sent'] += packet_length
            flow['timestamps'].append(timestamp.timestamp())
            flow['packet_lengths'].append(packet_length)

            # Match request and response times
            if src_ip.startswith("192.168") or src_ip.startswith("10."):  # Example: Local IP
                flow['timestamps'].append(timestamp.timestamp())
            else:
                if flow['timestamps']:
                    request_time = flow['timestamps'].pop(0)
                    response_time = timestamp.timestamp()
                    flow['response_times'].append(response_time - request_time)

            # Compute flow duration
            duration = max(flow['timestamps']) - min(flow['timestamps']) if len(flow['timestamps']) > 1 else 0
            flow_sent_rate = flow['bytes_sent'] / duration if duration > 0 else 0
            flow_received_rate = flow['bytes_sent'] / duration if duration > 0 else 0  # Placeholder

            # Calculate packet length statistics
            lengths = flow['packet_lengths']
            packet_length_mean = statistics.mean(lengths)
            packet_length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
            packet_length_stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0
            packet_length_median = statistics.median(lengths)
            packet_length_mode = statistics.mode(lengths)
            packet_length_skew_from_median = packet_length_mean - packet_length_median
            packet_length_skew_from_mode = packet_length_mean - packet_length_mode
            packet_length_coeff_var = packet_length_stdev / packet_length_mean if packet_length_mean > 0 else 0

            columns = [
                "SourceIP", "DestinationIP", "SourcePort", "DestinationPort", "Duration",
                "FlowBytesSent", "FlowSentRate", "FlowBytesReceived", "FlowReceivedRate",
                "PacketLengthVariance", "PacketLengthStandardDeviation", "PacketLengthMean",
                "PacketLengthMedian", "PacketLengthMode", "PacketLengthSkewFromMedian",
                "PacketLengthSkewFromMode", "PacketLengthCoefficientofVariation",
                "PacketTimeVariance", "PacketTimeStandardDeviation", "PacketTimeMean",
                "PacketTimeMedian", "PacketTimeMode", "PacketTimeSkewFromMedian",
                "PacketTimeSkewFromMode", "PacketTimeCoefficientofVariation",
            ]
            data = [[
                src_ip, dst_ip, src_port, dst_port, duration,
                flow['bytes_sent'], flow_sent_rate, flow['bytes_sent'], flow_received_rate,
                packet_length_variance, packet_length_stdev, packet_length_mean,
                packet_length_median, packet_length_mode, packet_length_skew_from_median,
                packet_length_skew_from_mode, packet_length_coeff_var, 0,0,0,0,0,0,0,0
            ]]

            df = pd.DataFrame(data, columns=columns)
            data = prepare_data(df)

            prediction = rf_model.predict(data)
            print(f"Prediction for packet: {prediction}")

        except Exception as e:
            print(f"Error processing packet: {e}")
            continue

