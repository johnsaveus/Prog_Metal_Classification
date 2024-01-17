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
            if path.endswith('Train'):     
                pattern = r'..\\raw_data\\Train\\(.*?)(?:\\|$)'
            elif path.endswith('Test'):
                pattern = r'..\\raw_data\\Test\\(.*?)(?:\\|$)'
            elif path.endswith('val'):
                pattern = r'..\\raw_data\\val\\(.*?)(?:\\|$)'
            match = re.search(pattern,path_to_riff)
            label = match.group(1) 
            label_list.append(label)

    return riff_list, label_list