import pyshark

def capture_packets(interface, packet_queue, port):
    print(f"Starting capture on interface: {interface}, port: {port}")
    capture = pyshark.LiveCapture(
        interface=interface,
        bpf_filter=f"port {port}",
    )
    
    try:
        for packet in capture.sniff_continuously():
            packet_queue.put(packet)
    except KeyboardInterrupt:
        print("Stopping capture...")
        #packet_queue.put(None)
    except Exception as e:
        print(f"An error occurred during packet capture: {e}")
    finally: 
        print("Finalizing capture...")
        if capture:
            capture.close()  # Assegura que la captura es tanca correctament
        packet_queue.put(None)  # Notifica que la captura ha acabat

