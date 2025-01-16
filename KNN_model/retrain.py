import pandas as pd
import joblib
from sklearn.utils import resample
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

def ip_to_numeric(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return None  # Per manejar IPs mal formades

knn_model = joblib.load("./knn.joblib")
new_data = pd.read_csv('./doh_decrypted.csv', sep =",")

new_data = new_data.drop(columns=['TimeStamp'])
new_data = new_data.rename(columns={
    'ResponseTimeVariance': 'ResponseTimeTimeVariance',
    'ResponseTimeStandardDeviation': 'ResponseTimeTimeStandardDeviation',
    'ResponseTimeMean': 'ResponseTimeTimeMean',
    'ResponseTimeMedian': 'ResponseTimeTimeMedian',
    'ResponseTimeMode': 'ResponseTimeTimeMode',
    'ResponseTimeSkewFromMedian': 'ResponseTimeTimeSkewFromMedian',
    'ResponseTimeSkewFromMode': 'ResponseTimeTimeSkewFromMode',
    'ResponseTimeCoefficientofVariation': 'ResponseTimeTimeCoefficientofVariation'
})
new_data['SourceIP'] = new_data['SourceIP'].apply(ip_to_numeric)
new_data['DestinationIP'] = new_data['DestinationIP'].apply(ip_to_numeric)
new_data = new_data.apply(pd.to_numeric, errors='coerce').fillna(0)
#Shuffle data
new_data = new_data.sample(frac=1,random_state=42).reset_index(drop=True)

shuffled_features = new_data.drop(columns=['Classification'])
shuffled_labels = new_data['Classification']

#Separe training and test
train_data, test_data, labels_train, labels_test = train_test_split(
    shuffled_features, shuffled_labels, test_size=0.33, random_state=42)

df_train = pd.concat([train_data, labels_train], axis=1)

df_majority = df_train[df_train['Classification'] == 0]
df_minority = df_train[df_train['Classification'] == 1]

# Downsample majority class (zeros) in the training set
df_majority_downsampled = resample(df_majority, 
    replace=False,    # Do not sample with replacement
    n_samples=len(df_train)//2,  # Match minority class size
    random_state=42)  # Ensure reproducibility

# Upsample minority class (ones) in the training set
df_minority_upsampled = resample(df_minority, 
    replace=True,     # Sample with replacement
    n_samples=len(df_train)//3,  # Match majority class size
    random_state=42)  # Ensure reproducibility

df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])

df_balanced = df_balanced.sample(frac=1,random_state=42).reset_index(drop=True)

balanced_data = df_balanced.drop(columns=['Classification'])
balanced_labels = df_balanced['Classification']

#balanced_data = balanced_data.fillna(0)  # Substituir per 0
knn_model.fit(balanced_data, balanced_labels)

predictionsKNN = knn_model.predict(test_data)
knn_accuracy = accuracy_score(labels_test, predictionsKNN)
precision = precision_score(labels_test, predictionsKNN)
recall = recall_score(labels_test, predictionsKNN)
f1 = f1_score(labels_test, predictionsKNN)

print(f"Accuracy K-NN: {knn_accuracy}")
print(f"Presicion K-NN: {precision}")
print(f"Recall K-NN: {recall}")
print(f"F1 Score: {f1}")

joblib.dump(knn_model, 'knnRetrained.joblib')