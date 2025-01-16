import pyshark

def capture_packets(interface, ssl_keylog_file, packet_queue):
    print(f"Starting capture on interface: {interface}, port: 443")
    capture = pyshark.LiveCapture(
        interface=interface,
        display_filter='tcp.port==443 or tcp.port==80',
        override_prefs={"tls.keylog_file": ssl_keylog_file}
    )
    try:
        for packet in capture.sniff_continuously():
           packet_queue.put(packet)
    except KeyboardInterrupt:
        print("Stopping capture...")
        packet_queue.put(None)

