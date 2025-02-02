from doh_decryptor.capture import capture_packets
from queue import Queue
from threading import Thread
from doh_decryptor.packetAnalyzer import  PacketAnalyzer

def main():
    interface = "eth0"  # Replace with your network interface
    ssl_keylog_file = "~/sslkeys.log"  # SSL key log file
    output_file = "doh_decrypted_traffic.csv"
    packetAnalyzer = PacketAnalyzer(output_file=output_file)

    packet_queue = Queue()
    processor_thread = Thread(target=packetAnalyzer.process_packets, args=(packet_queue,))
    processor_thread.start()

    capture_packets(interface, ssl_keylog_file, packet_queue)
    processor_thread.join()


if __name__ == "__main__":
    main()


