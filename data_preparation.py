import os
import re
import numpy as np
import pandas as pd
import librosa
from librosa import feature
from sklearn.preprocessing import StandardScaler

'''     The original raw data need to be cleaned
        We make a list. Each element contains the path to each riff at [0] and the label (band) at [1]
''' 

def raw_to_list(path):
    riff_list = []
    label_list = []
    for bands, albums, riffs in os.walk(path):
        for riff in riffs:
            path_to_riff = os.path.join(bands,riff)
            # Pattern to match the band which is the labels (It is path specific)
            pattern = r'raw_data\\(.*?)(?:\\|$)' 
            match = re.search(pattern,path_to_riff)
            label = match.group(1)
            riff_list.append(path_to_riff)
            label_list.append(label)

    return riff_list , label_list

'''     Function to extract short term features for our wav files
        Cite "https://github.com/farzanaanjum/Music-Genre-Classification-with-Python/blob/master/Music_genre_classification.ipynb"
        The features that are used are 1)zero crossing rate, 2)chroma stft, 3)spectral centroid +++++++++++++ !!!!!

'''

def wav_featurize(wav_list):

    csv_matrix = []

    for wav in wav_list:

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
                               mfcc]
        )
        
        mean_features = [np.mean(feat) for feat in feature_vector]
        csv_matrix.append(mean_features)

    scaled_matrix = scale_features(csv_matrix)
    df = pd.DataFrame(scaled_matrix)

    return df 

""" Need to scale the features cause some features have values 0.01... and others 5000..
"""
def scale_features(input_data):
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_data)

    return scaled_input


def main():
    data_path = 'raw_data'
    wavs , labels = raw_to_list(data_path)
    frame = wav_featurize(wavs)
    frame.to_csv('clean_data\csv_data',index=False)

if __name__ == '__main__':
    main()

