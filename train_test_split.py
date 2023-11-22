import pandas as pd


''' For each class(band) there are 40 instances.
    Those 40 instances are comprised of 8 albums -> 5 instances per album
    From those 5 instances we will use 3 for training, 1 for validation and 1 for test
    The 8th album of each band will always be used as a independent test dataset
'''

def dataset_split(path_to_data: str):

    df = pd.read_csv(path_to_data)
    train_csv = pd.DataFrame(columns=df.columns)
    test_csv = pd.DataFrame(columns=df.columns)
    
    for i in range(0, len(df), 5):
        train_data = df.iloc[i+2:i+5]
        test_data = df.iloc[i:i + 2]

        train_csv = pd.concat([train_csv,train_data],ignore_index=True)
        test_csv = pd.concat([test_csv,test_data],ignore_index=True)

    train_csv.to_csv(r'clean_data\train',index=False)
    test_csv.to_csv(r'clean_data\test',index=False)

    return None

if __name__ == '__main__':
    main('clean_data\csv_data')
