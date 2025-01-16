# DoH Traffic Classification Research

## Introduction

As the internet continues to evolve, encryption has become essential for protecting user privacy and data. DNS over HTTPS (DoH) is one such advancement, designed to enhance privacy by encrypting DNS queries within HTTPS traffic, protecting users from DNS hijacking and surveillance. However, this also introduces challenges for network operators, security analysts, and researchers who need to monitor and classify traffic for network management, anomaly detection, and security purposes.

The ability to classify DoH traffic accurately is crucial for several reasons:

- **Network Security:** Identifying DoH traffic is important for detecting potential misuse or hidden communications, especially in security-sensitive environments such as corporate networks or government infrastructures.
- **Performance Monitoring:** Classifying DoH traffic helps network administrators understand traffic load on their networks and optimize performance, particularly in environments where encrypted traffic is prevalent.
- **Policy Enforcement:** Organizations and ISPs may need to monitor and manage DoH traffic to enforce security policies, control the use of encrypted DNS, or ensure compliance with local regulations.

Accurate classification of DoH traffic is crucial for managing modern encrypted networks, enabling enhanced security and performance without compromising privacy. DNS over HTTPS (DoH) was first introduced by Mozilla in 2018 and later adopted by Chrome. Due to its privacy-enhancing features, DoH has garnered significant attention in the research community.

Several studies have explored detecting DoH traffic using machine learning techniques. Vekshin et al. explored detection through machine learning, while more advanced methods such as deep learning were applied by Fesl and Casanova, achieving high accuracy. Jerabek et al. combined IP tracking, machine learning, and active probing to detect DoH traffic, although challenges remain for standard systems.

In this project, we aim to train a classifier that uses accessible toolsets to detect DoH traffic with reasonable accuracy.

## Objectives

### General Objective

Develop a system to accurately classify DoH traffic from other types of encrypted traffic (e.g., HTTPS), maintaining user privacy while providing critical insights into network behavior.

### Specific Objectives

- Identify key flow characteristics that allow classification of DoH traffic.
- Implement and train machine learning models for traffic classification.
- Evaluate the system's accuracy using existing datasets and real traffic captures.
- Deploy a system that enables real-time classification of traffic in a simulated network.

## Methodology

### DoH Research and Analysis

Investigate specific characteristics and behaviors of DoH traffic compared to traditional DNS and HTTPS traffic. Identify flow-level features (e.g., packet size, flow duration, bytes transferred) that can distinguish DoH traffic from other encrypted traffic.

### Dataset Selection and Preprocessing

Use existing datasets such as:
- **CIRA-CIC-DoHBrw-2020**: Specific to DoH.
- **ISCX VPN-nonVPN Traffic Dataset** and **MAWI Working Group Traffic Archive**: Containing encrypted traffic like HTTPS and DNS.

Clean and label the data to ensure accurate representation of DoH and non-DoH traffic.

### Flow Feature Extraction

Use tools like **CICFlowMeter** or **Argus** to extract key traffic features, such as flow duration, packet size, and bytes transferred. These features will be essential for feeding the machine learning models.

### Training and Evaluation of Machine Learning Models

Train supervised machine learning models to classify DoH traffic based on the extracted flow data. Initial models may include Random Forest, Gradient Boosting, and Support Vector Machines (SVM). Based on the results, explore more advanced techniques like deep learning (e.g., Neural Networks).

### Testing

Set up a test environment where real-time traffic will be captured using tools like **Tcpdump** or **Wireshark**. The captured traffic will be processed by the trained machine learning model to classify it as DoH or non-DoH. 

For real-time monitoring and visualization, design a dashboard using **Grafana** or **Plotly Dash**.

## Tools to Be Used

- **Traffic capture:** Tcpdump or Wireshark for real-time network traffic capture.
- **Flow extraction:** CICFlowMeter or Argus to extract flow features.
- **Machine learning models:** Scikit-learn for models like Random Forest and SVM. XGBoost for advanced classification, and Keras for Neural Networks if needed.
- **Datasets:** 
  - CIRA-CIC-DoHBrw-2020 (specific to DoH traffic).
  - ISCX VPN-nonVPN Traffic Dataset and MAWI Working Group Traffic Archive for general encrypted traffic (HTTPS).
- **Development environment:** Python for model implementation and training. Grafana or Plotly Dash for real-time traffic classification visualization.

## Expected Conclusions

This project aims to:

1. Demonstrate the feasibility of identifying and classifying DoH traffic with high accuracy using machine learning or flow traffic feature analysis.
2. Accurately classify DoH traffic by identifying flow-level characteristics that distinguish it from other types of encrypted traffic.
3. Contribute to the understanding of how encrypted traffic, particularly DoH, can be identified without decrypting the content, which is essential for maintaining user privacy while ensuring network security.
4. Highlight potential benefits such as:
   - **Improved Network Security:** Detect potential misuse or abnormal patterns like command-and-control (C2) communications or data exfiltration attempts.
   - **Operational Efficiency:** Help network administrators allocate resources efficiently, optimize traffic routing, and reduce latency.
   - **Data Privacy Compliance:** Enable network monitoring of encrypted traffic types without violating user privacy, aligning with privacy regulations like GDPR or CCPA.


## References

1. Greenwade, G. D. (1993). The Comprehensive TeX Archive Network (CTAN). *TUGBoat*, 14(3), 342–351.

2. Vekshin, D., Hynek, K., & Cejka, T. (2020). Doh insight: Detecting DNS over HTTPS by machine learning. In *Proceedings of the 15th International Conference on Availability, Reliability and Security* (pp. 1-8).

3. Jerabek, K., Hynek, K., Rysavy, O., & Burgetova, I. (2023). DNS over HTTPS detection using standard flow telemetry. *IEEE Access*, 11, 50000–50012. IEEE.

4. Casanova, L. F. G., & Lin, P.-C. (2021). Generalized classification of DNS over HTTPS traffic with deep learning. In *2021 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* (pp. 1903-1907). IEEE.

5. Fesl, J., Konopa, M., & Jelínek, J. (2023). A novel deep-learning based approach to DNS over HTTPS network traffic detection. *International Journal of Electrical and Computer Engineering (IJECE)*, 13(6), 6691–6700.

# DoH Traffic Classification Implementation 

## Demo and the Commands

Install the project requirements: 
pip install -r requirements.txt

Set the interface you'd like to capture on in the file Live Traffic\main.py 

Run the project from the root directory: 
python3 Live\ Traffic/main.py

To check the prediction from the CMD run the following curl commands : 
For example.com
curl -X GET "https://dns.google/dns-query?dns=AAABAAABAAAAAAAAB2V4YW1wbGUDY29tAAABAAE" \
  -H "Accept: application/dns-message"

For openai.com
curl -X GET "https://dns.google/dns-query?dns=AAABAAABAAAAAAAAB29wZW5haQNhY29tAAABAAE" \
  -H "Accept: application/dns-message"

For github.com
curl -X GET "https://dns.google/dns-query?dns=AAABAAABAAAAAAAAB2dpdGh1YgNjb20AAAEAAQ" \
  -H "Accept: application/dns-message"

## Project Structure 
In this project we've trained two different models, KNN and random forest. The main model used in the demo and the project is in the ".\Models folder" but the other trained model can be found in ".\KNN_model" for further backup and adaptability. 
The main code of the project is in ".\Live Traffic" where traffic is captured live from the set interface and specific features corresponding to the model are being extracted by ".\Live Traffic\doh_decryptor\flow.py".
After the features are extracted from the live data, the model is applied on the features and prediction is done by ".\Live Traffic\doh_decryptor\packetAnalyzer.py". After the prediction, if DoH detected, the ip table rules are updated to drop the packet from being routed to the LAN. 

In the end the folder ".\Tools" contains toolset developed throughout the project but not used in the demo. You can find automatic browsing by bash, python, and packet labeling by decrypting the TLS packets. 
