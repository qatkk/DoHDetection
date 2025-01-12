import pandas as pd
import joblib 
import ipaddress
from datetime import datetime
import numpy 
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

#Functions

# Function to convert ips to numeric numbers
def ip_to_numeric(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return None  # Per manejar IPs mal formades

# Function to convert timestamp to float
def timestamp_to_float(timestamp):
    try:
        dt_object = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        return dt_object.timestamp()
    except ValueError:
        return None  
    
#Main
combined_df = pd.read_csv('./DataSets/all.csv', sep=",") 
labels = pd.array(combined_df['DoH'])
print (f"the number of DoH packets are {numpy.count_nonzero(labels == True)}")
print(f"number of total packets are {len(labels)}")

#DROPS
combined_df = combined_df.drop(columns=['TimeStamp'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeVariance'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeStandardDeviation'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeMean'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeMedian'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeMode'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeSkewFromMedian'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeSkewFromMode'])
combined_df = combined_df.drop(columns=['ResponseTimeTimeCoefficientofVariation'])

shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# After shuffling, separate the features and labels again
shuffled_features = shuffled_df.drop(columns=['DoH'])
shuffled_labels = shuffled_df['DoH']

# Convert string features to floats
shuffled_features['SourceIP'] = shuffled_features['SourceIP'].apply(ip_to_numeric)
shuffled_features['DestinationIP'] = shuffled_features['DestinationIP'].apply(ip_to_numeric)

# Convert string values can't convert to float to NaN
shuffled_features = shuffled_features.apply(pd.to_numeric, errors='coerce')

# Check if any NaN value
if shuffled_features['SourceIP'].isna().any() or shuffled_features['DestinationIP'].isna().any(): #or shuffled_features['TimeStamp'].isna().any():
    print("Hi ha valors no convertibles a numèric. Files amb NaN:")
    print(shuffled_features[shuffled_features.isna().any(axis=1)])
    shuffled_features = shuffled_features.dropna()
# If any NaN, convert it to 0s
shuffled_features = shuffled_features.fillna(0)

# Initialize KFold (5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=False)
knn_accuracies = []
random_forest_accuracies = []
# Create the 5 folds and print the train and test sets for each fold
fold_number = 1

rf = RandomForestClassifier(max_depth=15, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

for train_index, test_index in kf.split(shuffled_features):
    # Get train and test sets for this fold
    X_train, X_test = shuffled_features.iloc[train_index], shuffled_features.iloc[test_index]
    y_train, y_test = shuffled_labels.iloc[train_index], shuffled_labels.iloc[test_index]
    
    print(f"Fold {fold_number}:")
    print(f"Training Set Size: {len(X_train)}")
    print(f"Test Set Size: {len(X_test)}")

    # Separate the majority and minority classes in the training set
    df_train = pd.concat([X_train, y_train], axis=1)
    df_majority = df_train[df_train['DoH'] == 0]
    df_minority = df_train[df_train['DoH'] == 1]
    
    print(f"Original class distribution in training data:\n{df_train['DoH'].value_counts()}")

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
    if fold_number == 1 :
        print(f"the length of the minority after upsampling is {len(df_minority_upsampled)} and the majority is {len(df_majority_downsampled)}")
    # Combine the upsampled minority class with the downsampled majority class in the training set
    
    print(f"Distribution of the majority class before downsampling: {df_majority.shape[0]}")
    print(f"Distribution of the minority class before upsampling: {df_minority.shape[0]}")

    combined_train = pd.concat([df_majority_downsampled, df_minority_upsampled])

    print(f"Class distribution after resampling:\n{combined_train['DoH'].value_counts()}")

    # Shuffle the training data
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate the features and labels in the resampled training set
    X_train_resampled = combined_train.drop(columns=['DoH'])
    y_train_resampled = combined_train['DoH']

    #  time to do the training 

    # Train the k-NN classifier on the resampled training set
    knn.fit(X_train_resampled, y_train_resampled)
    
    # Predict the labels for the test set (no resampling for test set)
    y_pred = knn.predict(X_test)

    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)

    print("k-NN Classification Report:")
    print(classification_report(y_test, y_pred))

    # Append the accuracy to the list for tracking the overall performance
    knn_accuracies.append(accuracy)

    # Train the Random Forest classifier on the resampled training set
    rf.fit(X_train_resampled, y_train_resampled)
    
    # Predict the labels for the test set (no resampling for test set)
    y_pred_rf = rf.predict(X_test)

    # Calculate accuracy for Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    random_forest_accuracies.append(accuracy_rf)

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    fold_number += 1

print (f" the accuracies of the k-nn is {numpy.round(knn_accuracies,4)} with average of {numpy.mean(knn_accuracies):.4f}")
print (f" the accuracies of the random forest is {numpy.round(random_forest_accuracies,4)} with average of {numpy.mean(random_forest_accuracies):.4f}")

joblib.dump(rf, 'rf.joblib')
joblib.dump(knn, 'knn.joblib')