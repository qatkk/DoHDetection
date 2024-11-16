import pandas as pd 
import numpy 
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

combined_df = pd.read_csv('./DataSets/combined.csv', sep=";") 
labels = pd.array(combined_df['is_doh'])
print (f"the number of DoH packets are {numpy.count_nonzero(labels == 1)}")
print(f"number of total packets are {len(labels)}")

combined_df = combined_df.drop(columns=['datasrc'])
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# After shuffling, separate the features and labels again
shuffled_features = shuffled_df.drop(columns=['is_doh'])
shuffled_labels = shuffled_df['is_doh']

# Initialize KFold (5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=False)
knn_accuracies = []
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
    df_majority = df_train[df_train['is_doh'] == 0]
    df_minority = df_train[df_train['is_doh'] == 1]
    
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
    X_train_resampled = combined_train.drop(columns=['is_doh'])
    y_train_resampled = combined_train['is_doh']

    #  time to do the training 
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the k-NN classifier on the resampled training set
    knn.fit(X_train_resampled, y_train_resampled)
    
    # Predict the labels for the test set (no resampling for test set)
    y_pred = knn.predict(X_test)

    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)

    # Append the accuracy to the list for tracking the overall performance
    knn_accuracies.append(accuracy)

    rf = RandomForestClassifier(max_depth=15, random_state=42)

    # Train the Random Forest classifier on the resampled training set
    rf.fit(X_train_resampled, y_train_resampled)
    
    # Predict the labels for the test set (no resampling for test set)
    y_pred_rf = rf.predict(X_test)

    # Calculate accuracy for Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    random_forest_accuracies.append(accuracy_rf)

    fold_number += 1

print (f" the accuracies of the k-nn is {numpy.round(knn_accuracies,4)} with average of {numpy.mean(knn_accuracies):.4f}")
print (f" the accuracies of the random forest is {numpy.round(random_forest_accuracies,4)} with average of {numpy.mean(random_forest_accuracies):.4f}")
