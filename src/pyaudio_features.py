import os
import re
import pandas as pd
import numpy as np
import librosa
import pyAudioAnalysis.ShortTermFeatures as sF
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_import(path:str):
    '''Original data need cleaning. This function extracting riff(wav)
    path and band name for labelng'''
    riff_list = []
    label_list = []
    for bands, albums, riffs in os.walk(path):
        for riff in riffs:
            path_to_riff = os.path.join(bands,riff)
            relative_path = os.path.relpath(path_to_riff, path)
            band_name = relative_path.split(os.path.sep)[0]
            riff_list.append(path_to_riff)
            label_list.append(band_name)
    return riff_list, label_list

def pyaudio_featurize(y_array):

    feature_matrix = []
    sr = 22050
    for y in y_array:
        feature_vector = []
        feats, names = sF.feature_extraction(y,sr, 0.06*sr, 0.03*sr,deltas=True)
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

def label_encoding(labels):
    encoder = LabelEncoder()    
    return encoder.fit_transform(labels)

def merge_labels(df,labels):
    df.insert(1,'Band',label_encoding(labels=labels),True)

def trim_wav(wavs):
    sr = 22050
    duration = 10
    trimmed_y = []
    for wav in wavs:
        y, _ = librosa.load(wav)
        samples = int(duration*sr)
        y_remain = y[:samples]
        trimmed_y.append(y_remain)
    return trimmed_y