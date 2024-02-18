import os
import re
import pandas as pd
import numpy as np
import librosa
import pyAudioAnalysis.ShortTermFeatures as sF
from sklearn.preprocessing import  LabelEncoder

def data_import(path:str):
    '''
    Original data need cleaning. This function is extracting riff (wav)
    path and class label based on the raw input data file
    '''
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
    '''
    Using PyAudioAnalysis [1] for feature extraction. Specifically Short term feature extraction with deltas.
    Each feature is then used to compute its mean and standard deviation that comprise the final feature vector.
    Feature Vector shape for each instance is 136.
    [1] https://github.com/tyiannak/pyAudioAnalysis
    '''
    feature_matrix = []
    # Equal signal rate = 22050 for all songs
    sr = 22050
    for y in y_array:
        feature_vector = []
        # Setting frame size to 60ms and a frame step of 30ms (50% overlap)
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
    # Encode band names: str -> integer ('Band A'=0, 'Band B'=1 , etc...)
    encoder = LabelEncoder()    
    return encoder.fit_transform(labels)

def merge_labels(df,labels):
    # Insert labels into a dataframe
    df.insert(1,'Band',label_encoding(labels=labels),True)

def trim_wav(wavs):
    '''Most riffs are +10 seconds. This function is setting an upper bound time of 10 seconds per riff.
    There are few cases were riffs are <10 sec. Not taken into account'''
    sr = 22050
    duration = 10
    trimmed_y = []
    for wav in wavs:
        y, _ = librosa.load(wav)
        samples = int(duration*sr)
        y_remain = y[:samples]
        trimmed_y.append(y_remain)
        
    return trimmed_y