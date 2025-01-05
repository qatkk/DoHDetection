# DoHDetection

The same procedure as the paper "DoH Insight: Detecting DNS over HTTPS by Machine Learning" has been applied to their dataset synthesized in chrome. After doing so the accuracies for a 5-NN and a random forest was 99.23 and 99.62 respectively. These values are close to the ones reported in the paper (99.6 and 99.9) but less which is reasonable due to the training being done on lesser amount of data. 


### How to capture labeled packets: 

In order to create our own synthetic data, we will run fire fox with DoH enabled and decrypt packets to label our datasets. For doing so first run the command below and in the same tab run firefox.

export SSLKEYLOGFILE=/path/to/project/DoHDetection/sslkeys.log
