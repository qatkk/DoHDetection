from enum import Enum
from typing import Any
from doh_decryptor.context.packet_flow_key import get_packet_flow_key
from doh_decryptor.features.flow_bytes import FlowBytes
from doh_decryptor.features.packet_length import PacketLength
from doh_decryptor.features.packet_time import PacketTime
from doh_decryptor.features.response_times import ResponseTime
from doh_decryptor import  constants
import numpy

class Flow:
    def __init__(self, packet: Any, direction: Enum):
        self.dest_ip, self.src_ip, self.src_port, self.dest_port = get_packet_flow_key(packet, direction)
        self.packets = []
        self.latest_timestamp = 0
        self.start_timestamp = 0

    def add_packet(self, packet: Any, direction: Enum) -> None:
        self.packets.append((packet, direction))
        self.latest_timestamp = max(packet.sniff_time.timestamp(), self.latest_timestamp)
        if self.start_timestamp == 0:
            self.start_timestamp = packet.sniff_time.timestamp()

    def get_data(self) -> dict:
        """This method obtains the values of the features extracted from each flow.

        Returns:
            dict: Dictionary of flow features with cleaned data types.
        """
        flow_bytes = FlowBytes(self)
        packet_length = PacketLength(self)
        packet_time = PacketTime(self)
        response = ResponseTime(self)

        # Get raw data
        data = {
            # Basic IP information (already clean types)
            'SourceIP': self.src_ip,
            'DestinationIP': self.dest_ip,
            'SourcePort': self.src_port,
            'DestinationPort': self.dest_port,

            # Basic information from packet times
            'TimeStamp': packet_time.get_time_stamp(),
            'Duration': float(packet_time.get_duration()),

            # Information from the amount of bytes (convert to integers)
            'FlowBytesSent': int(flow_bytes.get_bytes_sent()),
            'FlowSentRate': float(flow_bytes.get_sent_rate()),
            'FlowBytesReceived': int(flow_bytes.get_bytes_received()),
            'FlowReceivedRate': float(flow_bytes.get_received_rate()),

            # Statistical info obtained from Packet lengths (convert all to float)
            'PacketLengthVariance': float(packet_length.get_var()),
            'PacketLengthStandardDeviation': float(packet_length.get_std()),
            'PacketLengthMean': float(packet_length.get_mean()),
            'PacketLengthMedian': float(packet_length.get_median()),
            'PacketLengthMode': float(packet_length.get_mode()),
            'PacketLengthSkewFromMedian': float(packet_length.get_skew()),
            'PacketLengthSkewFromMode': float(packet_length.get_skew2()),
            'PacketLengthCoefficientofVariation': float(packet_length.get_cov()),

            # Statistical info obtained from Packet times (convert all to float)
            'PacketTimeVariance': float(packet_time.get_var()),
            'PacketTimeStandardDeviation': float(packet_time.get_std()),
            'PacketTimeMean': float(packet_time.get_mean()),
            'PacketTimeMedian': float(packet_time.get_median()),
            'PacketTimeMode': float(packet_time.get_mode()),
            'PacketTimeSkewFromMedian': float(packet_time.get_skew()),
            'PacketTimeSkewFromMode': float(packet_time.get_skew2()),
            'PacketTimeCoefficientofVariation': float(packet_time.get_cov()),

            # Response Time (convert all to float)
            'ResponseTimeTimeVariance': float(response.get_var()),
            'ResponseTimeTimeStandardDeviation': float(response.get_std()),
            'ResponseTimeTimeMean': float(response.get_mean()),
            'ResponseTimeTimeMedian': float(response.get_median()),
            'ResponseTimeTimeMode': float(response.get_mode()),
            'ResponseTimeTimeSkewFromMedian': float(response.get_skew()),
            'ResponseTimeTimeSkewFromMode': float(response.get_skew2()),
            'ResponseTimeTimeCoefficientofVariation': float(response.get_cov()),

            'DoH': bool(self.is_doh()),
        }

        # Clean any potential NaN or infinite values
        for key, value in data.items():
            if isinstance(value, float):
                if not isinstance(value, float) or numpy.isnan(value) or numpy.isinf(value):
                    data[key] = -1.0

        return data



    def get_duration(self) -> float:
        return self.latest_timestamp - self.start_timestamp

    def is_doh(self) -> bool:
        return self.src_ip in constants.DOH_IPS or self.dest_ip in constants.DOH_IPS
