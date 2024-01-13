import pandas as pd
import numpy as np
import librosa
from librosa import feature
from sklearn.preprocessing import LabelEncoder

def feauturizer(wavs,labels):

    csv_matrix = []
    hp = 3000  
    for wav in wavs:   
        y, _ = librosa.load(wav)
        feature_vector = []

        feature_vector.extend([feature.zero_crossing_rate(y,hop_length=hp),
                               feature.rms(y=y,hop_length=hp),
                               feature.spectral_centroid(y=y,hop_length=hp),
                               feature.spectral_rolloff(y=y,hop_length=hp),
                               feature.mfcc(y=y,hop_length=hp)])
        
        mean_features = [np.mean(feat) for feat in feature_vector]
        csv_matrix.append(mean_features)

    df_features = pd.DataFrame(csv_matrix, columns = ['zero_crossing_rate',
                                                      'rms',
                                                      'spectral_centroid',
                                                      'spectral_rolloff',
                                                      'mfcc'])
    df_features.insert(-1,'Band',label_encoding(labels=labels),True)
        
    return df_features
    
def label_encoding(labels):
    encoder = LabelEncoder()

    return encoder.fit_transform(labels)
    