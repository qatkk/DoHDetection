import pyshark
from scapy.all import IP, TCP, send
from joblib import load
from process import process_packets

#Load model
model = load("RandomForest_features.joblib")

def extract_features(packet):
    features = process_packets(packet)
    return features

def process_packet(packet_queue):
    while True:
        packet = packet_queue.get()
        #features = extract_features(packet)

        #Model prediction
        #prediction = model.predict(features)
        prediction = True

        if prediction: #If it's DOH
            #Send RST packet to destination
           rst_pkt_dst = IP(src=packet.ip.src, dst=packet.ip.dst) / \
                          TCP(sport=int(packet.tcp.srcport), dport=int(packet.tcp.dstport), flags="R", seq=int(packet.tcp.seq) + 1)
           send(rst_pkt_dst, verbose=0)
           
           rst_pkt_src = IP(src=packet.ip.dst, dst=packet.ip.src) / \
                          TCP(sport=int(packet.tcp.dstport), dport=int(packet.tcp.srcport), flags="R", seq=int(packet.tcp.ack))
           send(rst_pkt_src, verbose=0)
           
           print("Conection ended.")
