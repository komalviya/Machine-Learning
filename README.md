# International Vibes
Machine Learning project by Jason Boyle, Komal Malviya & Garbhan Power

The aim of this project is to develop a machine learning system to be able to classify music samples based on the region from which they originate. The trained model should be able to take a sample of traditional music from a specific culture/region (e.g. traditional Irish music, traditional Middle-Eastern music etc.) and be able to correctly identify which culture/region the music sample came from.

## Data Preprocessing
The accumulated music samples were converted to .wav format and resampled at 22kHz (This was done outside of python)
The wav_file_30_second_split.py program is used to divide each musical sample into 30 second snippets before feature extraction.

## Feature Extraction
The audio_featre_extraction.py program uses the Librosa audio processing library for python was to extract a set of spectral (frequency-domain) and rhythmic features from each 30 second audio snippet. Features were selected based on their ability to characterise different aspects of musical data including instrument identification and overall musical rhythm. These features include:
 * Zero Crossing Rate
 * Spectral Contrast
 * Spectral Bandwidth
 * Spectral Flatness
 * Spectral Rolloff
 * Tempogram
 * Mel Frequency Cepstral Coefficients (MFCC)

## kNN Model
The current model utilises the 2 nearest neighbours to make the prediction
and uses the "distance" weights option. The parameters were selected through:
1) Plotting the accuracy of the model vs the number of neighbours
 (see "uniform accuracy.png" and "distance accuracy.png")
2) Using 5-Fold cross-validation to plot the MSE mean and variance for 
different number of neighbours (see all other pics lol)
