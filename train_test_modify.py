import pandas as pd


''' For each class(band) there are 40 instances.
    Those 40 instances are comprised of 8 albums -> 5 instances per album
    From those 5 instances we will use 3 for training, 1 for validation and 1 for test
    The 8th album of each band will always be used as a independent test dataset
'''

def main(path: str):

    train_csv = pd.read_csv(path + '\\train')
    test_csv = pd.read_csv(path + '\\test')

    removed = []
    for i in range(0, len(train_csv), 5):
        selected_test = train_csv.iloc[i:i+1]
        removed.append(i)
        test_csv = pd.concat([test_csv,selected_test],axis=0,ignore_index=True)

    train_csv.drop(removed,inplace=True)
    train_csv.reset_index(inplace=True)

    train_csv.to_csv(r'clean_data\train',index=False)
    test_csv.to_csv(r'clean_data\test',index=False)

    return None

if __name__ == '__main__':
    main('clean_data')
