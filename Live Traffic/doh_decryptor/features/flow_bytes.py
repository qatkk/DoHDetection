from doh_decryptor.context.packet_direction import PacketDirection
from doh_decryptor.features.packet_time import PacketTime


class FlowBytes:
    def __init__(self, feature):
        self.feature = feature

    def direction_list(self) -> list:
        feat = self.feature
        return [direction.name for i, (_, direction) in enumerate(feat.packets) if i < 50]

    def get_bytes_sent(self) -> int:
        feat = self.feature
        return sum(int(packet.length) for packet, direction in feat.packets if direction == PacketDirection.FORWARD)

    def get_sent_rate(self) -> float:
        sent = self.get_bytes_sent()
        duration = PacketTime(self.feature).get_duration()
        return sent / duration if duration > 0 else -1

    def get_bytes_received(self) -> int:
        feat = self.feature
        return sum(int(packet.length) for packet, direction in feat.packets if direction == PacketDirection.REVERSE)

    def get_received_rate(self) -> float:
        received = self.get_bytes_received()
        duration = PacketTime(self.feature).get_duration()
        return received / duration if duration > 0 else -1

    def get_forward_header_bytes(self) -> int:
        def header_size(packet):
            header_length = int(packet.ip.hdr_len)  # IP header length in bytes
            if hasattr(packet, 'tcp'):
                header_length += int(packet.tcp.hdr_len)
            elif hasattr(packet, 'udp'):
                header_length += 8  # UDP header size is fixed at 8 bytes
            return header_length

        feat = self.feature
        return sum(header_size(packet) for packet, direction in feat.packets if direction == PacketDirection.FORWARD)

    def get_forward_rate(self) -> float:
        forward = self.get_forward_header_bytes()
        duration = PacketTime(self.feature).get_duration()
        return forward / duration if duration > 0 else -1

    def get_reverse_header_bytes(self) -> int:
        def header_size(packet):
            header_length = int(packet.ip.hdr_len)
            if hasattr(packet, 'tcp'):
                header_length += int(packet.tcp.hdr_len)
            elif hasattr(packet, 'udp'):
                header_length += 8
            return header_length

        feat = self.feature
        return sum(header_size(packet) for packet, direction in feat.packets if direction == PacketDirection.REVERSE)

    def get_reverse_rate(self) -> float:
        reverse = self.get_reverse_header_bytes()
        duration = PacketTime(self.feature).get_duration()
        return reverse / duration if duration > 0 else -1

    def get_header_in_out_ratio(self) -> float:
        reverse_header_bytes = self.get_reverse_header_bytes()
        forward_header_bytes = self.get_forward_header_bytes()
        return forward_header_bytes / reverse_header_bytes if reverse_header_bytes != 0 else -1

    def get_initial_ttl(self) -> int:
        feat = self.feature
        return int(feat.packets[0][0].ip.ttl)
