from doh_decryptor.context.packet_direction import PacketDirection

def get_packet_flow_key(packet, direction) -> tuple:
    if hasattr(packet, 'tcp'):
        protocol = 'tcp'
    elif hasattr(packet, 'udp'):
        protocol = 'udp'
    else:
        raise Exception('Only TCP and UDP protocols are supported.')

    if direction == PacketDirection.FORWARD:
        dest_ip = packet.ip.dst
        src_ip = packet.ip.src
        src_port = getattr(packet[protocol], 'srcport', None)
        dest_port = getattr(packet[protocol], 'dstport', None)
    elif direction == PacketDirection.REVERSE:
        dest_ip = packet.ip.src
        src_ip = packet.ip.dst
        src_port = getattr(packet[protocol], 'dstport', None)
        dest_port = getattr(packet[protocol], 'srcport', None)
    else:
        raise ValueError('Invalid packet direction.')

    return dest_ip, src_ip, src_port, dest_port

