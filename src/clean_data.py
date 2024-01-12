import os
import re

def raw_to_list(path:str):
    '''Original data need cleaning. This function extracting riff(wav)
    path and band name for labelng'''
    riff_list = []
    label_list = []
    for bands, albums, riffs in os.walk(path):
        for riff in riffs:
            path_to_riff = os.path.join(bands,riff)
            riff_list.append(path_to_riff)
            if path.endswith('train'):     
                pattern = r'..\\raw_data\\train\\(.*?)(?:\\|$)'
            elif path.endswith('test'):
                pattern = r'..\\raw_data\\test\\(.*?)(?:\\|$)'
            match = re.search(pattern,path_to_riff)
            label = match.group(1) 
            label_list.append(label)

    return riff_list, label_list