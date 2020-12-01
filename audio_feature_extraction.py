import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from playsound import playsound
import csv

# Labels
#1 = Traditional_Irish
#2 = Indian_Madhyalaya
#3 = Brazillian_Samba
#4 = Japanese_Koto

# Writing the header line to the .csv file
file = open('C:/Machine_Learning_Audio_Samples/Japanese_Koto_Audio_Data.csv', 'w', newline='')
header = 'zero_crossing_rate spectral_centroid spectral_contrast spectral_bandwidth spectral_flatness spectral_rolloff tempo '
for i in range(0,20):
    header += f'mfcc[{i}] '
header += 'label '
writer = csv.writer(file)
writer.writerow(header.split())

# Extracting Features from the music sample
for n in range(0,120):
    sample = f'C:/Machine_Learning_Audio_Samples/{n}_Japanese_Koto.wav'
    x, sr = librosa.load(sample, sr = None) 

    zero_cross_rate = librosa.feature.zero_crossing_rate(x)
    spect_centroid = librosa.feature.spectral_centroid(x)
    spect_contrast = librosa.feature.spectral_contrast(x)
    spect_bandwidth = librosa.feature.spectral_bandwidth(x)
    spect_flatness = librosa.feature.spectral_flatness(x)
    spect_rolloff = librosa.feature.spectral_rolloff(x)
    tempo = librosa.feature.tempogram(x)
    mfcc = librosa.feature.mfcc(x)

    # Writing the Features to the csv file
    file = open('C:/Machine_Learning_Audio_Samples/Japanese_Koto_Audio_Data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        features = f'{np.mean(zero_cross_rate)} {np.mean(spect_centroid)} {np.mean(spect_contrast)} {np.mean(spect_bandwidth)} {np.mean(spect_flatness)} {np.mean(spect_rolloff)} {np.mean(tempo)} '
        for i in mfcc:
            features += f'{np.mean(i)} '
        
        writer.writerow(features.split())

print("done")