from collections import defaultdict
from doh_decryptor.context.packet_flow_key import get_packet_flow_key
from doh_decryptor.context.packet_direction import PacketDirection
from doh_decryptor.flow import Flow

EXPIRED_UPDATE = 40


def is_expired(packet, flow):
    return (packet.sniff_time.timestamp() - flow.latest_timestamp) > EXPIRED_UPDATE


def is_expired_flow(flow):
    return flow.duration > 90


def save_flow(flow):
    data = flow.get_data()
    # Implement your save logic here


class FlowManager:
    def __init__(self):
        self.flows = {}

    def add_packet_to_flow(self, packet, direction=PacketDirection.FORWARD):
        # First try to find an existing flow with forward direction
        packet_flow_key = get_packet_flow_key(packet, PacketDirection.FORWARD)
        flow = self.flows.get(packet_flow_key)

        if flow is None:
            # If no forward flow exists, try reverse direction
            reverse_flow_key = get_packet_flow_key(packet, PacketDirection.REVERSE)
            flow = self.flows.get(reverse_flow_key)

            if flow is None:
                # No existing flow found, create new one with original direction
                flow = Flow(packet, direction)
                self.flows[packet_flow_key] = flow
            else:
                # Found a reverse flow, use reverse direction for this packet
                direction = PacketDirection.REVERSE

        # Check if the flow is expired
        if is_expired(packet, flow):
            # Create new flow with original direction
            flow = Flow(packet, direction)
            # Use the appropriate key based on direction
            key = get_packet_flow_key(packet, direction)
            self.flows[key] = flow

        flow.add_packet(packet, direction)
        return flow.get_data()

    def get_flows(self):
        return list(self.flows.values())

    def process_flows(self):
        for key, flow in list(self.flows.items()):
            if is_expired_flow(flow):
                save_flow(flow)
                del self.flows[key]