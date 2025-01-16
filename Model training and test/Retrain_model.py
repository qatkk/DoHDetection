import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import load, dump
from sklearn.utils import resample, shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import ipaddress

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '../Models/New_RandomForest.joblib')
dataset_path = os.path.join(current_dir, '../Datasets/doh_decrypted_traffic_4.csv')


rf_model = load(model_path)

new_dataset = pd.read_csv(dataset_path, sep=",")

new_dataset['SourceIP'] = new_dataset['SourceIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
new_dataset['DestinationIP'] = new_dataset['DestinationIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
new_dataset = new_dataset.rename(columns={'ResponseTimeCoefficientofVariation': 'ResponseTimeTimeCoefficientofVariation'})
new_dataset = new_dataset.rename(columns={'ResponseTimeMean': 'ResponseTimeTimeMean'})
new_dataset = new_dataset.rename(columns={'ResponseTimeMedian': 'ResponseTimeTimeMedian'})
new_dataset = new_dataset.rename(columns={'ResponseTimeMode': 'ResponseTimeTimeMode'})
new_dataset = new_dataset.rename(columns={'ResponseTimeSkewFromMedian': 'ResponseTimeTimeSkewFromMedian'})
new_dataset = new_dataset.rename(columns={'ResponseTimeSkewFromMode': 'ResponseTimeTimeSkewFromMode'})
new_dataset = new_dataset.rename(columns={'ResponseTimeStandardDeviation': 'ResponseTimeTimeStandardDeviation'})
new_dataset = new_dataset.rename(columns={'ResponseTimeVariance': 'ResponseTimeTimeVariance'})

X = new_dataset.drop(columns=['Classification', 'TimeStamp'])
X = X.fillna(0)
y = new_dataset['Classification']

X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_majority = df_train[df_train['Classification'] == 0]
df_minority = df_train[df_train['Classification'] == 1]

df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # Do not sample with replacement
                                   n_samples=len(df_train)//2,  # Match minority class size
                                   random_state=42)

df_minority_upsampled = resample(df_minority, 
                                    replace=True,     # Sample with replacement
                                    n_samples=len(df_train)//3,  # Match majority class size
                                    random_state=42)  # Ensure reproducibility

combined_train = pd.concat([df_majority_downsampled, df_minority_upsampled])

combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_resampled = combined_train.drop(columns=['Classification'])
y_train_resampled = combined_train['Classification']

rf_model.fit(X_train_resampled, y_train_resampled)

predictions = rf_model.predict(X_test)
print(f"Random forest predictions:{predictions}")
print(pd.Series(predictions).value_counts())

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)  
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Resultados
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

dump(rf_model, model_path)
