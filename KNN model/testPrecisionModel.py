import joblib
import pandas as pd
import ipaddress
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Function to convert ips to numeric numbers
def ip_to_numeric(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return None

# Load dataset
real_data = pd.read_csv('./d.csv', sep =",")

# Rename, drops and normalizing data
real_data = real_data.drop(columns=['TimeStamp'])
real_data = real_data.rename(columns={
    'ResponseTimeVariance': 'ResponseTimeTimeVariance',
    'ResponseTimeStandardDeviation': 'ResponseTimeTimeStandardDeviation',
    'ResponseTimeMean': 'ResponseTimeTimeMean',
    'ResponseTimeMedian': 'ResponseTimeTimeMedian',
    'ResponseTimeMode': 'ResponseTimeTimeMode',
    'ResponseTimeSkewFromMedian': 'ResponseTimeTimeSkewFromMedian',
    'ResponseTimeSkewFromMode': 'ResponseTimeTimeSkewFromMode',
    'ResponseTimeCoefficientofVariation': 'ResponseTimeTimeCoefficientofVariation'
})
real_data['SourceIP'] = real_data['SourceIP'].apply(ip_to_numeric)
real_data['DestinationIP'] = real_data['DestinationIP'].apply(ip_to_numeric)

# Convert string values can't convert to float to NaN
real_data = real_data.apply(pd.to_numeric, errors='coerce').fillna(0)

Y_true = real_data['Classification']
X_test = real_data.drop(columns=['Classification'])

knn_model = joblib.load("./knn.joblib")

# KNN predictions
predictionsKNN = knn_model.predict(X_test)
knn_accuracy = accuracy_score(Y_true, predictionsKNN)
precision = precision_score(Y_true, predictionsKNN)
recall = recall_score(Y_true, predictionsKNN)
f1 = f1_score(Y_true, predictionsKNN)
print(f"Accuracy K-NN: {knn_accuracy}")
print(f"Presicion K-NN: {precision}")
print(f"Recall K-NN: {recall}")
print(f"F1 Score: {f1}")

knn_false_count = (predictionsKNN == 0).sum()
knn_true_count = (predictionsKNN == 1).sum()
print(f"K-NN - False: {knn_false_count}, True: {knn_true_count}")