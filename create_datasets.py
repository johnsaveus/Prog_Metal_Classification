import os
import re
import numpy as np
import pandas as pd
import librosa
from librosa import feature
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings


warnings.simplefilter('ignore')

'''     The original raw data need to be cleaned
        We make a list. Each element contains the path to each riff at [0] and the label (band) at [1]
''' 

def raw_to_list(path:str):
    riff_list = []
    label_list = []
    for bands, albums, riffs in os.walk(path):
        for riff in riffs:
            path_to_riff = os.path.join(bands,riff)
            # Pattern to match the band which is the labels (It is path specific)
            if path.endswith('train'):     
                pattern = r'raw_data\\train\\(.*?)(?:\\|$)'
            elif path.endswith('test'):
                pattern = r'raw_data\\test\\(.*?)(?:\\|$)'
            match = re.search(pattern,path_to_riff)
            label = match.group(1)
            riff_list.append(path_to_riff)
            label_list.append(label)

    return riff_list , label_list

'''     Function to extract short term features for our wav files
        Cite "https://github.com/farzanaanjum/Music-Genre-Classification-with-Python/blob/master/Music_genre_classification.ipynb"
        The features that are used are 1)zero crossing rate, 2)chroma stft, 3)spectral centroid +++++++++++++ !!!!!

'''

def wav_featurize(wav_list,labels):

    csv_matrix = []

    for idx , wav in enumerate(wav_list):

        feature_vector = []

        y , sr = librosa.load(wav)
        zero_crossing_rate = feature.zero_crossing_rate(y,pad=False)
        chroma_stft = feature.chroma_stft(y=y,sr=sr)
        spectral_centroid = feature.spectral_centroid(y=y,sr=sr)
        spectral_bandwidth = feature.spectral_bandwidth(y=y,sr=sr)
        spectral_rolloff = feature.spectral_rolloff(y=y, sr=sr)
        mfcc = feature.mfcc(y=y,sr=sr)

        feature_vector.extend([zero_crossing_rate,
                               chroma_stft,
                               spectral_centroid,
                               spectral_bandwidth,
                               spectral_rolloff,
                               mfcc,
                               ]
        )
        
        mean_features = [np.mean(feat) for feat in feature_vector]
        csv_matrix.append(mean_features)

    df_features = pd.DataFrame(csv_matrix,columns=['zero_crossing_rate',
                                                   'chroma_stft',
                                                   'spectral_centroid',
                                                   'spectral_bandwidth',
                                                   'spectral_rolloff',
                                                   'mfcc'])
    df_endp = pd.DataFrame(labels,columns=['band_name'])

    df = pd.concat([df_features,df_endp], axis = 1, join='inner')

    df = label_encoding(df)

    return df


# The unique band names need to be converted to classes of integers
def label_encoding(frame):
        
    encoder = preprocessing.LabelEncoder()
    frame.iloc[:,-1:] = encoder.fit_transform(frame.iloc[:,-1:])

    return frame

def convert_to_pd(raw_path):

    wavs , labels = raw_to_list(raw_path)   
    frame = wav_featurize(wavs,labels)
    
    return frame

''' For each class(band) there are 40 instances.
    Those 40 instances are comprised of 8 albums -> 5 instances per album
    From those 5 instances we will use 3 for training, 1 for validation and 1 for test
    The 8th album of each band will always be used as a independent test dataset
'''

def split(raw_path: str):

    train_csv = convert_to_pd(raw_path + '\\train')
    test_csv = convert_to_pd(raw_path + '\\test')

    removed = []
    for i in range(0, len(train_csv), 5):
        selected_test = train_csv.iloc[i:i+1]
        removed.append(i)
        test_csv = pd.concat([test_csv,selected_test],axis=0,ignore_index=True)

    train_csv.drop(removed,inplace=True)
    train_csv.reset_index(inplace=True)

    train_csv.to_csv(r'yolo\train',index=False)
    test_csv.to_csv(r'yolo\test',index=False)

    return None

if __name__ == '__main__':
    split('raw_data')