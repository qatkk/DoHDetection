import statistics
import joblib
import pandas as pd
import ipaddress
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
            'response_times': [],
        })
        self.model = joblib.load('./knn.joblib')
        with open(self.output_file, 'w') as output:
            output.write("SourceIP,DestinationIP,SourcePort,DestinationPort,TimeStamp,Duration,"
                         "FlowBytesSent,FlowSentRate,FlowBytesReceived,FlowReceivedRate,"
                         "PacketLengthVariance,PacketLengthStandardDeviation,PacketLengthMean,"
                         "PacketLengthMedian,PacketLengthMode,PacketLengthSkewFromMedian,"
                         "PacketLengthSkewFromMode,PacketLengthCoefficientofVariation,"
                         "PacketTimeVariance,PacketTimeStandardDeviation,PacketTimeMean,"
                         "PacketTimeMedian,PacketTimeMode,PacketTimeSkewFromMedian,"
                         "PacketTimeSkewFromMode,PacketTimeCoefficientofVariation,"
                         "ResponseTimeTimeVariance,ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean,"
                         "ResponseTimeTimeMedian,ResponseTimeTimeMode,ResponseTimeTimeSkewFromMedian,"
                         "ResponseTimeTimeSkewFromMode,ResponseTimeTimeCoefficientofVariation,Classification\n")

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
        with open(self.output_file, 'a') as output:
            output.write(','.join(str(data[key]) for key in data.keys()) + '\n')
        
        df = pd.DataFrame([data])
        df= df.drop(columns=['TimeStamp'])
        df= df.drop(columns=['DoH'])
        
        df['SourceIP'] = df['SourceIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
        df['DestinationIP'] = df['DestinationIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
        
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        prediction = (self.model).predict(df)
        print(f"Prediction for packet is: {prediction}")
        
        with open('predictions.txt', 'w') as output:
            output.write('Prediction for packet:' + str(prediction) + '\n')


    def process_packets(self, packet_queue):
        while True:
            packet = packet_queue.get()
            if packet is None:
                break
            self.process_packet(packet)