import statistics
import pandas as pd
from joblib import load
from collections import defaultdict
from doh_decryptor.context.packet_direction import PacketDirection
from doh_decryptor.flow_session import FlowManager
from doh_decryptor.context.packet_flow_key import get_packet_flow_key  # Add this import


class PacketAnalyzer:
    def __init__(self, output_file):
        self.output_file = output_file
        self.flow_manager = FlowManager()
        self.flows = defaultdict(lambda: {
            'bytes_sent': 0,
            'bytes_received': 0,
            'timestamps': [],
            'packet_lengths': [],
            'response_times': []
        })

        self.model = load('RandomForest_features.joblib')

        #with open(self.output_file, 'w') as output:
        #    output.write("SourceIP,DestinationIP,SourcePort,DestinationPort,TimeStamp,Duration,"
        #                 "FlowBytesSent,FlowSentRate,FlowBytesReceived,FlowReceivedRate,"
        #                 "PacketLengthVariance,PacketLengthStandardDeviation,PacketLengthMean,"
        #                 "PacketLengthMedian,PacketLengthMode,PacketLengthSkewFromMedian,"
        #                 "PacketLengthSkewFromMode,PacketLengthCoefficientofVariation,"
        #                 "PacketTimeVariance,PacketTimeStandardDeviation,PacketTimeMean,"
        #                 "PacketTimeMedian,PacketTimeMode,PacketTimeSkewFromMedian,"
        #                 "PacketTimeSkewFromMode,PacketTimeCoefficientofVariation,"
        #                 "ResponseTimeVariance,ResponseTimeStandardDeviation,ResponseTimeMean,"
        #                 "ResponseTimeMedian,ResponseTimeMode,ResponseTimeSkewFromMedian,"
        #                 "ResponseTimeSkewFromMode,ResponseTimeCoefficientofVariation,Classification\n")

    def determine_direction(self, packet):
        forward_key = get_packet_flow_key(packet, PacketDirection.FORWARD)
        if forward_key in self.flow_manager.flows:
            return PacketDirection.FORWARD

        reverse_key = get_packet_flow_key(packet, PacketDirection.REVERSE)
        if reverse_key in self.flow_manager.flows:
            return PacketDirection.REVERSE

        return PacketDirection.FORWARD

    def process_packet(self, packet):
        direction = self.determine_direction(packet)
        data = self.flow_manager.add_packet_to_flow(packet, direction)


        df = pd.DataFrame([data])
        df[['SIP1', 'SIP2', 'SIP3', 'SIP4']] = df['SourceIP'].apply(lambda ip: pd.Series([int(octet) for octet in ip.split('.')]))
        df[['DIP1', 'DIP2', 'DIP3', 'DIP4']] = df['DestinationIP'].apply(lambda ip: pd.Series([int(octet) for octet in ip.split('.')]))
        label = df['DoH']
        df = df.drop(columns=['TimeStamp', 'DoH', 'SourceIP', 'DestinationIP'], axis=1)

        prediction = (self.model).predict(df)

        print(f"Prediction fot the packet is: {prediction}. In fact is {label}")

    def process_packets(self, packet_queue):
        while True:
            packet = packet_queue.get()
            if packet is None:
                break
            self.process_packet(packet)