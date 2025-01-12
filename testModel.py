from capture import capture_packets
from process import process_packets
from threading import Thread
from queue import Queue

interface = "Ethernet"  # Replace with your network interface

# Create a packet queue
packet_queue = Queue()

# Start packet processing thread
processor_thread = Thread(target=process_packets, args=(packet_queue,))
processor_thread.start()

# Start packet capture
capture_packets(interface, packet_queue, 443)

# Wait for processing to complete
processor_thread.join()
