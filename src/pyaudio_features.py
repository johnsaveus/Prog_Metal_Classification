import pandas as pd
import numpy as np
import librosa
import pyAudioAnalysis.ShortTermFeatures as sF
from sklearn.preprocessing import StandardScaler, LabelEncoder


def pyaudio_featurize(wavs):

    feature_matrix = []
    for wav in wavs:
        feature_vector = []
        y, sr = librosa.load(wav)
        feats, names = sF.feature_extraction(y,sr, 0.05*sr, 0.025*sr)
        feats_mean = np.mean(feats,axis=1)
        feats_std = np.std(feats,axis=1)
        feature_vector.extend(feats_mean)
        feature_vector.extend(feats_std)
        feature_matrix.append(feature_vector)
    names_mean = []
    names_std = []
    for name in names:
        names_mean.append(name+'_mean')
        names_std.append(name+'_std')
    feature_names = names_mean + names_std
    df_features = pd.DataFrame(feature_matrix, columns = feature_names)

    return df_features

def scaling(train,test):

    names = train.columns
    scaler = StandardScaler()
    scaler.fit(train)
    scaled_train = pd.DataFrame(scaler.transform(train),columns=names)
    scaled_test = pd.DataFrame(scaler.transform(test),columns=names)

    return scaled_train, scaled_test

def label_encoding(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

def merge_labels(df,labels):
    df.insert(68,'Band',label_encoding(labels=labels),True)
