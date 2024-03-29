import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

def calculate_metrics(class_report):
    '''
    Input must be classification report object from sklearn 
    Either train or test. This function is needed to extract single class metrics from classification report
    '''
    scores = {'precision':[],'recall':[],'f1_score':[]}
    for label in class_report:
        if label.isdigit():
            scores['precision'].append(class_report[label]['precision'])
            scores['recall'].append(class_report[label]['recall'])
            scores['f1_score'].append(class_report[label]['f1-score'])
    return scores

def cross_validate(model,dataset,folds=20):
    '''
    This is a custom cross validation function to train the models. We select
    randomly 200-50 train-test for each iteration. The classes on training 
    and test data are balanced. So there are 40 riffs for training and 10
    for test for each band. The metrics are calculated for each class individually
    and also f1 macro score for both sets
    '''
    np.random.seed(42)
    grouped = dataset.groupby('Band') # Creates different dfs with unique train_labels
    train_f1 = []
    test_f1 = []
    train_scores_all = {'precision':[],'recall':[],'f1_score':[]}
    test_scores_all = {'precision':[],'recall':[],'f1_score':[]}
    for _ in range(folds):
        test_ix = []
        for _, group in grouped:
            # Select 10 random samples from each class for testing. 20% of dataset
            sampling = group.sample(n=10) 
            # Indexes of those samples
            test_ix.extend(sampling.index.to_list())
        # Creating train-test data for each fold iteration
        train_set = dataset.drop(test_ix) # Drop test data
        test_set = dataset.iloc[test_ix] # Create test data
        X_train = train_set.drop(['Band'],axis=1)
        X_test = test_set.drop(['Band'],axis=1)
        y_train = train_set['Band']
        y_test = test_set['Band']
        # Scale training data (mean,std) and fit both training-test
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Train the model and make predictions for each fold
        model.fit(X_train,y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        # F1 macro score is used as the main metric
        train_f1.append(f1_score(y_train,train_preds,average='macro'))
        test_f1.append(f1_score(y_test,test_preds,average='macro'))
        # Calculate metrics for individual classes 
        cr_train = classification_report(y_train,train_preds,output_dict=True)
        cr_test = classification_report(y_test,test_preds,output_dict=True)
        train_metrics = calculate_metrics(cr_train)
        test_metrics = calculate_metrics(cr_test)

        train_scores_all['precision'].extend([train_metrics['precision']])
        train_scores_all['recall'].extend([train_metrics['recall']])
        train_scores_all['f1_score'].extend([train_metrics['f1_score']])

        test_scores_all['precision'].extend([test_metrics['precision']])
        test_scores_all['recall'].extend([test_metrics['recall']])
        test_scores_all['f1_score'].extend([test_metrics['f1_score']])
    
    # See avg_std_metrics function
    train_solo = avg_std_metrics(train_scores_all)
    test_solo = avg_std_metrics(test_scores_all)
    
    return [train_f1,train_solo], [test_f1,test_solo]


def avg_std_metrics(all_scores):
    # Calculates avg and std of metrics for all folds and each class seperately
    precision = np.array(all_scores['precision'])
    recall = np.array(all_scores['recall'])
    f1_score = np.array(all_scores['f1_score'])

    prec_avg, prec_std = np.mean(precision,axis=0), np.std(precision,axis=0)
    recall_avg, recall_std = np.mean(recall,axis=0), np.std(recall,axis=0)
    f1_avg, f1_std = np.mean(f1_score,axis=0), np.std(f1_score,axis=0)

    return {'precision':[prec_avg,prec_std],
            'recall':[recall_avg,recall_std],
            'f1_score':[f1_avg,f1_std]}    