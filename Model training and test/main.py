import pandas as pd 
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import ipaddress

combined_df = pd.read_csv('./Datasets/all.csv', sep=",") 
labels = pd.array(combined_df['DoH'])
print (f"the number of DoH packets are {np.count_nonzero(labels == 1)}")
print(f"number of total packets are {len(labels)}")

#Transform IPs into numerical values to treat them as features too
#combined_df[['SIP1', 'SIP2', 'SIP3', 'SIP4']] = combined_df['SourceIP'].apply(lambda ip: pd.Series([int(octet) for octet in ip.split('.')]))
#combined_df[['DIP1', 'DIP2', 'DIP3', 'DIP4']] = combined_df['DestinationIP'].apply(lambda ip: pd.Series([int(octet) for octet in ip.split('.')]))
combined_df['SourceIP'] = combined_df['SourceIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
combined_df['DestinationIP'] = combined_df['DestinationIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))

shuffled_features = combined_df.drop(columns=['DoH', 'TimeStamp'])
shuffled_features = shuffled_features.fillna(0)
shuffled_labels = combined_df['DoH']

print(shuffled_features.columns.to_list())

# Initialize KFold (5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=False)
random_forest_accuracies = []
# Create the 5 folds and print the train and test sets for each fold
fold_number = 1

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
    combined_train = pd.concat([df_majority_downsampled, df_minority_upsampled])

    # Shuffle the training data
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate the features and labels in the resampled training set
    X_train_resampled = combined_train.drop(columns=['DoH'])
    y_train_resampled = combined_train['DoH']

    #time to do the training 

    # Train the Random Forest classifier on the resampled training set
    rf = RandomForestClassifier(max_depth=15, random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    
    # Predict the labels for the test set (no resampling for test set)
    y_pred_rf = rf.predict(X_test)
    # Calculate accuracy for Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    random_forest_accuracies.append(accuracy_rf)

    fold_number += 1

dump(rf, './Models/RandomForest_fold.joblib')
print (f" the accuracies of the random forest is {np.round(random_forest_accuracies,4)} with average of {np.mean(random_forest_accuracies):.4f}")
