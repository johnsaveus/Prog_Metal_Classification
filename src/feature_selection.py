import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

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

def feature_target_corellation(dataset,threshold):
    corr_matrix = dataset.corr()
    features_to_drop = corr_matrix[abs(corr_matrix['Band']) < threshold].index
    dataset = dataset.drop(features_to_drop, axis=1)
    
    return dataset

    

    