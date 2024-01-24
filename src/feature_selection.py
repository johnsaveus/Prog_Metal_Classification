import pandas as pd
import numpy as np

def correlation_drop(dataset,threshold):

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