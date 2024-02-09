import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def features_corellation(dataset,threshold):

    correlation_matrix = dataset.corr()
    correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            correlation_value = correlation_matrix.iloc[i, j]
            if abs(correlation_value) > threshold :
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                correlated_pairs.append([feature1,feature2,correlation_value])

    first_feature = set()
    for cor in correlated_pairs:
        first_feature.add(cor[0])
    first_feature = list(first_feature)
    dataset = dataset.drop(first_feature,axis=1)

    return dataset, first_feature, correlated_pairs

def lasso_selection(model,dataset,folds=20):
    np.random.seed(42)
    grouped = dataset.groupby('Band') # Creates different dfs with unique train_labels
    f1 = []
    for fold in range(folds):
        test_dropped = []
        for _, group in grouped:
            sampling = group.sample(n=10) # Select 10 random samples for each train_label
            test_dropped.extend(sampling.index.to_list()) # Indexes of random test samples
        # Creating train-test data for each fold iteration
        train_set = dataset.drop(test_dropped) # Drop test data
        X_train = train_set.drop(['Band'],axis=1)
        y_train = train_set['Band']
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        model.fit(X_train,y_train)
        preds = model.predict(X_train)
        f1.append(f1_score(y_train,preds))
        

    