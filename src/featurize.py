import pandas as pd
import numpy as np
import librosa
from librosa import feature
from sklearn.preprocessing import StandardScaler, LabelEncoder

def featurizer(wavs):

    csv_matrix = []
    hp = 3000  
    for wav in wavs:   
        y, sr = librosa.load(wav)
        feature_vector = []

        feature_vector.extend([feature.zero_crossing_rate(y=y),
                               feature.rms(y=y),
                               feature.spectral_centroid(y=y,sr=sr),
                               feature.spectral_rolloff(y=y,sr=sr),
                               feature.spectral_bandwidth(y=y,sr=sr),
                               feature.spectral_contrast(y=y,sr=sr),
                               feature.spectral_flatness(y=y),
                               feature.mfcc(y=y,sr=sr),
                               feature.chroma_stft(y=y,sr=sr),
                               feature.chroma_cqt(y=y,sr=sr),
                               feature.chroma_cens(y=y,sr=sr)])

        mean_features = [np.mean(feat) for feat in feature_vector]
        #std_features = [np.std(feat) for feat in feature_vector]
        csv_matrix.append(mean_features)

    df_features = pd.DataFrame(csv_matrix, columns = ['zero_crossing_rate',
                                                      'rms',
                                                      'spectral_centroid',
                                                      'spectral_rolloff',
                                                      'spectral_bandwidth',
                                                      'spectral_contrast',
                                                      'spectral_flatness',
                                                      'mfcc',
                                                      'chroma_stft',
                                                      'chroma_cqt',
                                                      'chroma_cens'])
        
    return df_features
    
def label_encoding(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

'''def scaling(train,test):
    scaler = StandardScaler()
    scaler.fit(train)
    scaled_train = pd.DataFrame(scaler.transform(train))
    scaled_test = pd.DataFrame(scaler.transform(test))

    return scaled_train, scaled_test'''

def merge_labels(df,labels):
    df.insert(11,'Band',label_encoding(labels=labels),True)